package opencodego

import (
	"net/http"
	"testing"

	"github.com/enterpilot/gomodel/internal/core"
	"github.com/enterpilot/gomodel/internal/llmclient"
	"github.com/enterpilot/gomodel/internal/providers"
	"github.com/enterpilot/gomodel/internal/providers/providertest"
)

// newTestProvider builds a provider whose OpenAI-compatible and Anthropic
// /messages paths both point at the same test server.
func newTestProvider(serverURL string, client *http.Client) *Provider {
	return newHTTPTestProvider("sk-opencode", serverURL, client, llmclient.Hooks{})
}

func TestChatCompatibleContract(t *testing.T) {
	providertest.AssertChatCompatible(t, providertest.ChatCompatible{
		Registration:   Registration,
		Type:           "opencode_go",
		DefaultBaseURL: "https://opencode.ai/zen/go/v1",
		New: func(apiKey, baseURL string, client *http.Client, hooks llmclient.Hooks) core.Provider {
			opts := providertest.Options(hooks)
			opts.HTTPClient = client
			return New(providers.ProviderConfig{APIKey: apiKey, BaseURL: baseURL}, opts)
		},
		Embeddings: false,
	})
	providertest.AssertNoNativeSurfaces(t, newTestProvider("", nil))
}
