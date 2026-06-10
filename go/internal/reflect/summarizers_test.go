package reflect

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/nexusriot/ai-houkai/internal/memory"
)

func mems(texts ...string) []memory.Memory {
	out := make([]memory.Memory, len(texts))
	for i, t := range texts {
		out[i] = memory.Memory{Text: t, Type: memory.Episodic, Importance: 0.5}
	}
	return out
}

func TestBuildSummarizerExtractiveAliases(t *testing.T) {
	for _, spec := range []string{"", "extractive", "default", "none", "extractive:whatever"} {
		s, err := BuildSummarizer(spec, true)
		if err != nil {
			t.Fatalf("BuildSummarizer(%q): %v", spec, err)
		}
		got, err := s(context.Background(), mems("a fact", "another fact"))
		if err != nil {
			t.Fatalf("summarize: %v", err)
		}
		if !strings.HasPrefix(got, "[Reflection ×2] ") {
			t.Errorf("spec %q: got %q, want extractive output", spec, got)
		}
	}
}

func TestBuildSummarizerUnknownProvider(t *testing.T) {
	if _, err := BuildSummarizer("groq:llama3", true); err == nil {
		t.Error("expected error for unknown provider")
	}
}

func TestBuildSummarizerMissingModel(t *testing.T) {
	for _, spec := range []string{"ollama", "ollama:", "openai:  "} {
		if _, err := BuildSummarizer(spec, true); err == nil {
			t.Errorf("expected error for spec %q", spec)
		}
	}
}

func TestRenderPromptImportanceFirst(t *testing.T) {
	ms := []memory.Memory{
		{Text: "minor detail", Importance: 0.2},
		{Text: "key decision", Importance: 0.9},
		{Text: "medium fact", Importance: 0.5},
	}
	prompt := RenderPrompt(ms)
	for _, m := range ms {
		if !strings.Contains(prompt, "- "+m.Text) {
			t.Errorf("prompt missing %q", m.Text)
		}
	}
	if strings.Index(prompt, "key decision") > strings.Index(prompt, "minor detail") {
		t.Error("higher-importance memory should come first")
	}
}

// fakeOpenAICompat serves an OpenAI-style chat completions response and
// records the request.
func fakeOpenAICompat(t *testing.T, reply string, gotBody *map[string]any, gotPath *string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if gotPath != nil {
			*gotPath = r.URL.Path
		}
		if gotBody != nil {
			_ = json.NewDecoder(r.Body).Decode(gotBody)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"role": "assistant", "content": reply}},
			},
		})
	}))
}

func TestOllamaSummarizerCallsEndpoint(t *testing.T) {
	var body map[string]any
	var path string
	srv := fakeOpenAICompat(t, "A concise summary.", &body, &path)
	defer srv.Close()
	t.Setenv("OLLAMA_BASE_URL", srv.URL)

	s, err := BuildSummarizer("ollama:llama3.1:8b", false)
	if err != nil {
		t.Fatal(err)
	}
	got, err := s(context.Background(), mems("event one happened", "event two happened"))
	if err != nil {
		t.Fatal(err)
	}
	if got != "A concise summary." {
		t.Errorf("got %q", got)
	}
	if path != "/v1/chat/completions" {
		t.Errorf("path = %q", path)
	}
	// Model passed through verbatim, colons included.
	if body["model"] != "llama3.1:8b" {
		t.Errorf("model = %v", body["model"])
	}
}

func TestOllamaBaseURLWithV1Suffix(t *testing.T) {
	var path string
	srv := fakeOpenAICompat(t, "ok summary", nil, &path)
	defer srv.Close()
	t.Setenv("OLLAMA_BASE_URL", srv.URL+"/v1")

	s, _ := BuildSummarizer("ollama:m", false)
	if _, err := s(context.Background(), mems("x")); err != nil {
		t.Fatal(err)
	}
	if path != "/v1/chat/completions" {
		t.Errorf("path = %q (no double /v1 expected)", path)
	}
}

func TestFallbackOnError(t *testing.T) {
	t.Setenv("OLLAMA_BASE_URL", "http://127.0.0.1:1") // refused
	s, _ := BuildSummarizer("ollama:m", true)
	got, err := s(context.Background(), mems("fact a", "fact b"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(got, "[Reflection ×2] ") {
		t.Errorf("expected extractive fallback, got %q", got)
	}
}

func TestFallbackOnEmptyOutput(t *testing.T) {
	srv := fakeOpenAICompat(t, "   ", nil, nil)
	defer srv.Close()
	t.Setenv("OLLAMA_BASE_URL", srv.URL)
	s, _ := BuildSummarizer("ollama:m", true)
	got, err := s(context.Background(), mems("fact a", "fact b"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(got, "[Reflection ×2] ") {
		t.Errorf("expected extractive fallback, got %q", got)
	}
}

func TestNoFallbackErrors(t *testing.T) {
	t.Setenv("OLLAMA_BASE_URL", "http://127.0.0.1:1") // refused
	s, _ := BuildSummarizer("ollama:m", false)
	if _, err := s(context.Background(), mems("x")); err == nil {
		t.Error("expected error without fallback")
	}
}

func TestOutputStripped(t *testing.T) {
	srv := fakeOpenAICompat(t, "  trimmed summary \n", nil, nil)
	defer srv.Close()
	t.Setenv("OLLAMA_BASE_URL", srv.URL)
	s, _ := BuildSummarizer("ollama:m", true)
	got, err := s(context.Background(), mems("x"))
	if err != nil {
		t.Fatal(err)
	}
	if got != "trimmed summary" {
		t.Errorf("got %q", got)
	}
}

func TestOpenAIProvider(t *testing.T) {
	var auth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth = r.Header.Get("Authorization")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"content": "openai summary"}},
			},
		})
	}))
	defer srv.Close()
	t.Setenv("OPENAI_API_KEY", "sk-test")
	t.Setenv("OPENAI_BASE_URL", srv.URL)

	s, _ := BuildSummarizer("openai:gpt-4o-mini", false)
	got, err := s(context.Background(), mems("x"))
	if err != nil {
		t.Fatal(err)
	}
	if got != "openai summary" {
		t.Errorf("got %q", got)
	}
	if auth != "Bearer sk-test" {
		t.Errorf("auth = %q", auth)
	}
}

func TestAnthropicProvider(t *testing.T) {
	var key, version string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key = r.Header.Get("x-api-key")
		version = r.Header.Get("anthropic-version")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"content": []map[string]any{
				{"type": "text", "text": "anthropic "},
				{"type": "text", "text": "summary"},
				{"type": "tool_use", "text": "ignored"},
			},
		})
	}))
	defer srv.Close()
	t.Setenv("ANTHROPIC_API_KEY", "ak-test")
	t.Setenv("ANTHROPIC_BASE_URL", srv.URL)

	s, _ := BuildSummarizer("anthropic:claude-haiku-4-5", false)
	got, err := s(context.Background(), mems("x"))
	if err != nil {
		t.Fatal(err)
	}
	if got != "anthropic summary" {
		t.Errorf("got %q", got)
	}
	if key != "ak-test" || version == "" {
		t.Errorf("headers: key=%q version=%q", key, version)
	}
}

func TestMissingKeyErrorsWithoutFallback(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	s, _ := BuildSummarizer("openai:gpt-4o-mini", false)
	if _, err := s(context.Background(), mems("x")); err == nil {
		t.Error("expected error when API key missing")
	}
}
