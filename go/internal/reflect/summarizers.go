package reflect

// LLM-backed reflection summarizers, built from a config spec string.
//
// A summarizer is any Summarizer (see Engine). BuildSummarizer turns a
// spec string into one:
//
//	"extractive"                  built-in extractive default (no LLM)
//	"ollama:llama3.1"             local Ollama, OpenAI-compat endpoint
//	                              (OLLAMA_BASE_URL env)
//	"openai:gpt-4o-mini"          OpenAI chat completions (OPENAI_API_KEY)
//	"anthropic:claude-haiku-4-5"  Anthropic messages API (ANTHROPIC_API_KEY)
//
// The model part is passed through verbatim, so Ollama tags like
// "ollama:llama3.1:8b" work.
//
// By default the returned summarizer is wrapped with a fallback: if the LLM
// call fails (network down, key missing at call time, empty response) the
// built-in extractive summarizer is used instead and a warning is logged —
// reflection should degrade, not crash, when run unattended by the
// maintenance daemon.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/nexusriot/ai-houkai/internal/memory"
)

// Providers lists the accepted summarizer spec providers.
var Providers = []string{"extractive", "ollama", "openai", "anthropic"}

const (
	defaultOllamaBaseURL    = "http://localhost:11434"
	defaultOpenAIBaseURL    = "https://api.openai.com/v1"
	defaultAnthropicBaseURL = "https://api.anthropic.com"
	summarizerMaxTokens     = 1024
	summarizerTimeout       = 120 * time.Second
)

// PromptTemplate is the instruction wrapped around the events list.
const PromptTemplate = "You are condensing an AI agent's episodic memories into one durable " +
	"semantic memory.\n\nEvents (most important first):\n%s\n\n" +
	"Write a single concise summary (1-3 sentences) that captures the " +
	"durable facts, decisions, and patterns. Do not add information that " +
	"is not in the events. Output only the summary text."

// RenderPrompt formats a cluster of memories into the summarization prompt,
// most important first.
func RenderPrompt(memories []memory.Memory) string {
	ordered := make([]memory.Memory, len(memories))
	copy(ordered, memories)
	sort.SliceStable(ordered, func(i, j int) bool {
		return ordered[i].Importance > ordered[j].Importance
	})
	lines := make([]string, len(ordered))
	for i, m := range ordered {
		lines[i] = "- " + m.Text
	}
	return fmt.Sprintf(PromptTemplate, strings.Join(lines, "\n"))
}

var httpClient = &http.Client{Timeout: summarizerTimeout}

func postJSON(ctx context.Context, url string, headers map[string]string, payload any) (map[string]any, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("%s: HTTP %d: %s", url, resp.StatusCode, truncateStr(string(data), 200))
	}
	var out map[string]any
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, fmt.Errorf("%s: bad JSON: %w", url, err)
	}
	return out, nil
}

func truncateStr(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}

// chatCompletionsContent extracts choices[0].message.content from an
// OpenAI-compatible response.
func chatCompletionsContent(resp map[string]any) (string, error) {
	choices, _ := resp["choices"].([]any)
	if len(choices) == 0 {
		return "", fmt.Errorf("no choices in response")
	}
	choice, _ := choices[0].(map[string]any)
	msg, _ := choice["message"].(map[string]any)
	content, _ := msg["content"].(string)
	return content, nil
}

// ollamaSummarizer chats against Ollama's OpenAI-compatible endpoint.
func ollamaSummarizer(model string) Summarizer {
	return func(ctx context.Context, ms []memory.Memory) (string, error) {
		base := os.Getenv("OLLAMA_BASE_URL")
		if base == "" {
			base = defaultOllamaBaseURL
		}
		base = strings.TrimRight(base, "/")
		if !strings.HasSuffix(base, "/v1") {
			base += "/v1"
		}
		resp, err := postJSON(ctx, base+"/chat/completions", nil, map[string]any{
			"model":    model,
			"messages": []map[string]string{{"role": "user", "content": RenderPrompt(ms)}},
		})
		if err != nil {
			return "", err
		}
		return chatCompletionsContent(resp)
	}
}

func openaiSummarizer(model string) Summarizer {
	return func(ctx context.Context, ms []memory.Memory) (string, error) {
		key := os.Getenv("OPENAI_API_KEY")
		if key == "" {
			return "", fmt.Errorf("OPENAI_API_KEY is required for an 'openai:' summarizer")
		}
		base := os.Getenv("OPENAI_BASE_URL")
		if base == "" {
			base = defaultOpenAIBaseURL
		}
		resp, err := postJSON(ctx, strings.TrimRight(base, "/")+"/chat/completions",
			map[string]string{"Authorization": "Bearer " + key},
			map[string]any{
				"model":    model,
				"messages": []map[string]string{{"role": "user", "content": RenderPrompt(ms)}},
			})
		if err != nil {
			return "", err
		}
		return chatCompletionsContent(resp)
	}
}

func anthropicSummarizer(model string) Summarizer {
	return func(ctx context.Context, ms []memory.Memory) (string, error) {
		key := os.Getenv("ANTHROPIC_API_KEY")
		if key == "" {
			return "", fmt.Errorf("ANTHROPIC_API_KEY is required for an 'anthropic:' summarizer")
		}
		base := os.Getenv("ANTHROPIC_BASE_URL")
		if base == "" {
			base = defaultAnthropicBaseURL
		}
		resp, err := postJSON(ctx, strings.TrimRight(base, "/")+"/v1/messages",
			map[string]string{
				"x-api-key":         key,
				"anthropic-version": "2023-06-01",
			},
			map[string]any{
				"model":      model,
				"max_tokens": summarizerMaxTokens,
				"messages":   []map[string]string{{"role": "user", "content": RenderPrompt(ms)}},
			})
		if err != nil {
			return "", err
		}
		blocks, _ := resp["content"].([]any)
		var sb strings.Builder
		for _, b := range blocks {
			bm, _ := b.(map[string]any)
			if t, _ := bm["type"].(string); t == "text" {
				s, _ := bm["text"].(string)
				sb.WriteString(s)
			}
		}
		return sb.String(), nil
	}
}

// withFallback wraps an LLM summarizer so failures and empty outputs degrade
// to the extractive summarizer with a logged warning instead of erroring.
func withFallback(inner Summarizer, spec string) Summarizer {
	return func(ctx context.Context, ms []memory.Memory) (string, error) {
		text, err := inner(ctx, ms)
		if err != nil {
			log.Printf("ai-houkai: summarizer %q failed (%v) — falling back to extractive.", spec, err)
			return defaultSummarizer(ctx, ms)
		}
		text = strings.TrimSpace(text)
		if text == "" {
			log.Printf("ai-houkai: summarizer %q returned empty output — falling back to extractive.", spec)
			return defaultSummarizer(ctx, ms)
		}
		return text, nil
	}
}

// BuildSummarizer builds a Summarizer from a "provider:model" spec string.
//
// An empty spec, "extractive", "default" or "none" returns the built-in
// extractive summarizer. Otherwise the provider must be one of
// ollama / openai / anthropic with a non-empty model. When fallback is
// true (the usual case), LLM failures degrade to the extractive
// summarizer with a logged warning instead of erroring.
func BuildSummarizer(spec string, fallback bool) (Summarizer, error) {
	switch spec {
	case "", "extractive", "default", "none":
		return defaultSummarizer, nil
	}

	provider, model, _ := strings.Cut(spec, ":")
	provider = strings.ToLower(strings.TrimSpace(provider))
	model = strings.TrimSpace(model)

	if provider == "extractive" {
		return defaultSummarizer, nil
	}
	known := false
	for _, p := range Providers {
		if p == provider {
			known = true
			break
		}
	}
	if !known {
		return nil, fmt.Errorf("unknown summarizer provider %q in spec %q — expected one of %s",
			provider, spec, strings.Join(Providers, ", "))
	}
	if model == "" {
		return nil, fmt.Errorf("summarizer spec %q is missing a model — expected e.g. '%s:MODEL'",
			spec, provider)
	}

	var inner Summarizer
	switch provider {
	case "ollama":
		inner = ollamaSummarizer(model)
	case "openai":
		inner = openaiSummarizer(model)
	default:
		inner = anthropicSummarizer(model)
	}

	if fallback {
		return withFallback(inner, spec), nil
	}
	return inner, nil
}
