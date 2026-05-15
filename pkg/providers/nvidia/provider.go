// PicoClaw - Ultra-lightweight personal AI agent
// License: MIT
//
// Copyright (c) 2026 PicoClaw contributors

package nvidia

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/sipeed/picoclaw/pkg/providers/protocoltypes"
)

// Provider implements LLMProvider for NVIDIA API
type Provider struct {
	apiKey       string
	apiBase      string
	httpClient   *http.Client
	userAgent    string
	proxy        string
	timeout      time.Duration
	extraBody    map[string]any
	customHeaders map[string]string
}

// NewProvider creates a new NVIDIA API provider
func NewProvider(apiKey, apiBase, proxy, userAgent string, timeout float64, extraBody map[string]any, customHeaders map[string]string) *Provider {
	if apiBase == "" {
		apiBase = "https://integrate.api.nvidia.com/v1"
	}
	if userAgent == "" {
		userAgent = "PicoClaw/1.0"
	}

	httpTimeout := 60 * time.Second
	if timeout > 0 {
		httpTimeout = time.Duration(timeout) * time.Second
	}

	return &Provider{
		apiKey:       apiKey,
		apiBase:      apiBase,
		httpClient:   &http.Client{Timeout: httpTimeout},
		userAgent:    userAgent,
		proxy:        proxy,
		timeout:      httpTimeout,
		extraBody:    extraBody,
		customHeaders: customHeaders,
	}
}

// Chat implements the LLMProvider interface
func (p *Provider) Chat(
	ctx context.Context,
	messages []protocoltypes.Message,
	tools []protocoltypes.ToolDefinition,
	model string,
	options map[string]any,
) (*protocoltypes.LLMResponse, error) {
	// Build request payload
	payload := map[string]any{
		"model":    model,
		"messages": convertMessages(messages),
		"stream":   false,
	}

	// Add optional parameters from options
	if maxTokens, ok := options["max_tokens"]; ok {
		payload["max_tokens"] = maxTokens
	}
	if temperature, ok := options["temperature"]; ok {
		payload["temperature"] = temperature
	}
	if topP, ok := options["top_p"]; ok {
		payload["top_p"] = topP
	}

	// Merge extra body parameters
	if p.extraBody != nil {
		for k, v := range p.extraBody {
			if _, exists := payload[k]; !exists {
				payload[k] = v
			}
		}
	}

	// Marshal payload to JSON
	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal payload: %w", err)
	}

	// Create HTTP request
	url := p.apiBase + "/chat/completions"
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.apiKey)
	req.Header.Set("User-Agent", p.userAgent)

	// Add custom headers
	if p.customHeaders != nil {
		for k, v := range p.customHeaders {
			req.Header.Set(k, v)
		}
	}

	// Execute request
	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("http request: %w", err)
	}
	defer resp.Body.Close()

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read body: %w", err)
	}

	// Check for HTTP errors
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API returned %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var result struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}

	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("unmarshal response: %w", err)
	}

	if len(result.Choices) == 0 {
		return nil, fmt.Errorf("no choices in response")
	}

	finishReason := "stop"
	if len(result.Choices) > 0 && result.Choices[0].FinishReason != "" {
		finishReason = result.Choices[0].FinishReason
	}

	return &protocoltypes.LLMResponse{
		Content:      result.Choices[0].Message.Content,
		FinishReason: finishReason,
		Usage: &protocoltypes.UsageInfo{
			PromptTokens:     result.Usage.PromptTokens,
			CompletionTokens: result.Usage.CompletionTokens,
			TotalTokens:      result.Usage.TotalTokens,
		},
	}, nil
}

// GetDefaultModel returns the default NVIDIA model
func (p *Provider) GetDefaultModel() string {
	return "nvidia/llama-3.1-70b-instruct"
}

// convertMessages converts protocoltypes messages to NVIDIA API format
func convertMessages(messages []protocoltypes.Message) []map[string]string {
	result := make([]map[string]string, len(messages))
	for i, msg := range messages {
		result[i] = map[string]string{
			"role":    msg.Role,
			"content": msg.Content,
		}
	}
	return result
}
