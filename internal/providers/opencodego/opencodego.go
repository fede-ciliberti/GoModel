// Package opencodego provides OpenCode Zen (Go subscription) integration for
// the LLM gateway.
package opencodego

import (
	"context"
	"io"
	"net/http"
	"strings"

	"github.com/enterpilot/gomodel/internal/core"
	"github.com/enterpilot/gomodel/internal/providers"
	"github.com/enterpilot/gomodel/internal/providers/anthropic"
	"github.com/enterpilot/gomodel/internal/providers/deepseek"
	"github.com/enterpilot/gomodel/internal/providers/openai"
)

// defaultBaseURL is the OpenCode Zen "Go" endpoint. Its /chat/completions,
// /responses and /models routes are OpenAI-compatible (Bearer auth); its
// /messages route is Anthropic-native (x-api-key auth). Override with
// OPENCODE_GO_BASE_URL.
const defaultBaseURL = "https://opencode.ai/zen/go/v1"

// Registration provides factory registration for the OpenCode Go provider.
// The "opencode_go" type derives OPENCODE_GO_API_KEY, OPENCODE_GO_BASE_URL, and
// OPENCODE_GO_MODELS by convention.
var Registration = providers.Registration{
	Type: "opencode_go",
	New:  New,
	Discovery: providers.DiscoveryConfig{
		DefaultBaseURL: defaultBaseURL,
	},
}

// messagesProvider is the subset of the Anthropic provider OpenCode Go reuses
// to translate canonical chat requests onto the Anthropic-native /messages
// endpoint.
type messagesProvider interface {
	ChatCompletion(context.Context, *core.ChatRequest) (*core.ChatResponse, error)
	StreamChatCompletion(context.Context, *core.ChatRequest) (io.ReadCloser, error)
}

// Provider implements the core.Provider interface for OpenCode Go.
//
// OpenCode Zen exposes three upstream dialects behind one base URL, and
// GET /models names the one each model is served on natively in its "api"
// member (see routes.go): OpenAI-style /chat/completions, OpenAI-style
// /responses, or Anthropic-style /messages. That listing is the only source of
// the per-model routing — no model is listed in code — and every request follows
// it in both directions: the embedded ChatCompatible handles chat and translates
// Responses to chat, a chat request for a Responses-only model is translated to
// the Responses API and rendered back, a Responses request for a model Zen
// serves there is forwarded natively, and the models Zen serves on /messages are
// delegated to an Anthropic provider pinned to the same base URL, mirroring how
// GoModel already serves /v1/messages for native Anthropic. Every path
// normalizes to the canonical OpenAI-shaped response, so callers see one
// consistent surface.
type Provider struct {
	*openai.ChatCompatible
	messages messagesProvider
	// routes holds the API each model is served on, as GET /models publishes
	// it.
	routes *modelRoutes
	// adaptChat is the same hook ChatCompatible applies to chat requests. The
	// Responses-only route bypasses CompatibleProvider.ChatCompletion, so it
	// must apply the hook itself: the upstream rejects requests that carry no
	// reasoning parameter and needs the flat reasoning_effort shape.
	adaptChat func(*core.ChatRequest) (*core.ChatRequest, error)
	// adaptResponses is the same reasoning policy for a request forwarded
	// natively to /responses, which reaches neither AdaptChatRequest nor the
	// chat translation (see reasoning.go).
	adaptResponses func(*core.ResponsesRequest) *core.ResponsesRequest
	// headers yields the session/client identification headers (session.go).
	// Typed requests receive them through the ChatCompatible and Anthropic
	// hooks; Passthrough merges them under the caller's opaque headers.
	headers func(context.Context) http.Header
}

var _ core.Provider = (*Provider)(nil)

// New creates a new OpenCode Go provider.
func New(cfg providers.ProviderConfig, opts providers.ProviderOptions) core.Provider {
	baseURL := providers.ResolveBaseURL(cfg.BaseURL, defaultBaseURL)
	headers := requestHeaders(loadSessionHeaderEnabled())
	defaultEffort := loadDefaultReasoningEffort()
	adaptChat := chatRequestAdapter(defaultEffort, deepseek.LoadJSONSchemaMode())
	chat := openai.NewChatCompatible(cfg.APIKey, opts, compatibleConfig(baseURL, headers, adaptChat))
	// opts carries the shared keyring, so the /messages client rotates in step
	// with the chat client above rather than pinning the primary key.
	messages := anthropic.New(providers.ProviderConfig{APIKey: cfg.APIKey, APIKeys: cfg.APIKeys, BaseURL: baseURL}, opts)
	if native, ok := messages.(*anthropic.Provider); ok {
		native.SetRequestHeaders(headers)
	}
	return &Provider{
		ChatCompatible: chat,
		messages:       messages,
		routes:         &modelRoutes{},
		adaptChat:      adaptChat,
		adaptResponses: adaptResponsesRequest(defaultEffort),
		headers:        headers,
	}
}

// compatibleConfig describes the OpenAI-compatible half of the provider, which
// serves /chat/completions and /responses. The AdaptChatRequest hook carries
// OpenCode Zen's reasoning quirk (see reasoning.go) plus DeepSeek's rules for
// DeepSeek models (see deepseek.go); ChatRequestHeaders and
// ResponsesRequestHeaders carry the session/client identification headers (see
// session.go) on their own dialect, so a model reaches either endpoint with the
// same identification.
func compatibleConfig(baseURL string, headers func(context.Context) http.Header, adaptChat func(*core.ChatRequest) (*core.ChatRequest, error)) openai.CompatibleProviderConfig {
	return openai.CompatibleProviderConfig{
		ProviderName:            "opencode_go",
		BaseURL:                 baseURL,
		AdaptChatRequest:        adaptChat,
		ChatRequestHeaders:      chatRequestHeaders(headers),
		ResponsesRequestHeaders: responsesRequestHeaders(headers),
	}
}

// ChatCompletion routes each model to the API OpenCode Zen serves it on: the
// Anthropic /messages dialect for a Messages-only model, the Responses API for a
// Responses-only one, and /chat/completions for everything else.
func (p *Provider) ChatCompletion(ctx context.Context, req *core.ChatRequest) (*core.ChatResponse, error) {
	if req != nil {
		if p.messagesOnly(ctx, req.Model) {
			return p.messages.ChatCompletion(ctx, req)
		}
		if p.responsesOnly(ctx, req.Model) {
			return p.chatCompletionViaResponses(ctx, req)
		}
	}
	return p.ChatCompatible.ChatCompletion(ctx, req)
}

// chatCompletionViaResponses serves a chat request for a model OpenCode Zen
// answers only on /responses, rendering the result back as a chat completion.
func (p *Provider) chatCompletionViaResponses(ctx context.Context, req *core.ChatRequest) (*core.ChatResponse, error) {
	adapted, err := p.chatRequestForResponses(req)
	if err != nil {
		return nil, err
	}
	return openai.ChatViaResponses(ctx, p.Compatible(), adapted, p.ProviderName())
}

// chatRequestForResponses prepares a chat request for ChatViaResponses, which
// bypasses CompatibleProvider.ChatCompletion: it applies the chat adapter, then
// narrows an allowed_tools choice the way that path does, because the Responses
// translation rejects every choice that is not a plain function.
func (p *Provider) chatRequestForResponses(req *core.ChatRequest) (*core.ChatRequest, error) {
	adapted, err := p.adaptChat(req)
	if err != nil {
		return nil, err
	}
	if _, _, ok := core.AllowedToolsChoice(adapted.ToolChoice); !ok {
		return adapted, nil
	}
	tools, choice, err := core.NarrowToAllowedTools(adapted.Tools, adapted.ToolChoice)
	if err != nil {
		return nil, err
	}
	narrowed := *adapted
	narrowed.Tools = tools
	narrowed.ToolChoice = choice
	return &narrowed, nil
}

// StreamChatCompletion mirrors ChatCompletion's per-model endpoint routing.
func (p *Provider) StreamChatCompletion(ctx context.Context, req *core.ChatRequest) (io.ReadCloser, error) {
	if req != nil {
		if p.messagesOnly(ctx, req.Model) {
			return p.messages.StreamChatCompletion(ctx, req)
		}
		if p.responsesOnly(ctx, req.Model) {
			adapted, err := p.chatRequestForResponses(req)
			if err != nil {
				return nil, err
			}
			return openai.StreamChatViaResponses(ctx, p.Compatible(), adapted, p.ProviderName())
		}
	}
	return p.ChatCompatible.StreamChatCompletion(ctx, req)
}

// Responses forwards natively for a model OpenCode Zen serves on /responses,
// and translates to the model's own chat dialect otherwise, so /v1/responses
// honors the per-model routing in both directions.
func (p *Provider) Responses(ctx context.Context, req *core.ResponsesRequest) (*core.ResponsesResponse, error) {
	if req != nil && p.servesResponses(ctx, req.Model) {
		return p.Compatible().Responses(ctx, p.adaptResponses(req))
	}
	return providers.ResponsesViaChat(ctx, p, req, p.ProviderName())
}

// StreamResponses is Responses for streaming requests.
func (p *Provider) StreamResponses(ctx context.Context, req *core.ResponsesRequest) (io.ReadCloser, error) {
	if req != nil && p.servesResponses(ctx, req.Model) {
		return p.Compatible().StreamResponses(ctx, p.adaptResponses(req))
	}
	return providers.StreamResponsesViaChat(ctx, p, req, p.ProviderName())
}

// Passthrough forwards an opaque request with the identification headers
// filled in wherever the caller did not send its own. Passthrough is opaque
// by contract, so a caller-provided x-opencode-session, x-opencode-client, or
// User-Agent always wins over the gateway's value.
func (p *Provider) Passthrough(ctx context.Context, req *core.PassthroughRequest) (*core.PassthroughResponse, error) {
	if req == nil || p.headers == nil {
		return p.ChatCompatible.Passthrough(ctx, req)
	}
	merged := *req
	merged.Headers = withDefaultHeaders(req.Headers, p.headers(ctx))
	return p.ChatCompatible.Passthrough(ctx, &merged)
}

// withDefaultHeaders returns a copy of headers with every entry of defaults
// added that headers does not already carry. Names are compared
// case-insensitively so a non-canonical caller key still counts as present.
func withDefaultHeaders(headers, defaults http.Header) http.Header {
	merged := headers.Clone()
	if merged == nil {
		merged = make(http.Header, len(defaults))
	}
	for name, values := range defaults {
		if hasHeader(merged, name) {
			continue
		}
		merged[http.CanonicalHeaderKey(name)] = values
	}
	return merged
}

// hasHeader reports whether headers carries a non-empty entry for name under
// any casing of the key.
func hasHeader(headers http.Header, name string) bool {
	for key, values := range headers {
		if strings.EqualFold(key, name) && len(values) > 0 {
			return true
		}
	}
	return false
}

// Embeddings returns an error because OpenCode Go does not expose an embeddings endpoint.
func (p *Provider) Embeddings(_ context.Context, _ *core.EmbeddingRequest) (*core.EmbeddingResponse, error) {
	return nil, core.NewInvalidRequestError("opencode_go does not support embeddings", nil)
}
