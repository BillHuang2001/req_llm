defmodule ReqLLM.Providers.Deepseek do
  @moduledoc """
  DeepSeek AI provider – OpenAI-compatible Chat Completions API.

  ## Implementation

  Uses built-in OpenAI-style encoding/decoding defaults.
  DeepSeek is fully OpenAI-compatible, so no custom request/response handling is needed.

  ## Authentication

  Requires a DeepSeek API key from https://platform.deepseek.com/

  ## Configuration

      # Add to .env file (automatically loaded)
      DEEPSEEK_API_KEY=your-api-key

  ## Examples

      # Basic usage
      ReqLLM.generate_text("deepseek:deepseek-chat", "Hello!")

      # With custom parameters
      ReqLLM.generate_text("deepseek:deepseek-reasoner", "Write a function",
        temperature: 0.2,
        max_tokens: 2000
      )

      # Streaming
      ReqLLM.stream_text("deepseek:deepseek-chat", "Tell me a story")
      |> Enum.each(&IO.write/1)

      # JSON output (structured output)
      ReqLLM.generate_object("deepseek:deepseek-chat", "Extract user info",
        schema: [
          name: [type: :string, required: true],
          age: [type: :integer, required: true]
        ]
      )

      # With thinking mode enabled (default for reasoning models)
      ReqLLM.generate_text("deepseek:deepseek-v4-pro", "Solve this complex problem",
        provider_options: [
          thinking: %{type: "enabled"},
          reasoning_effort: :high
        ]
      )

      # With maximum reasoning effort for complex tasks
      ReqLLM.generate_text("deepseek:deepseek-v4-pro", "Complex reasoning task",
        provider_options: [
          reasoning_effort: :max
        ]
      )

      # Disable thinking mode
      ReqLLM.generate_text("deepseek:deepseek-v4-pro", "Quick question",
        provider_options: [
          thinking: %{type: "disabled"}
        ]
      )

  ## Models

  DeepSeek offers several models including:

  - `deepseek-chat` - General purpose conversational model
  - `deepseek-reasoner` - Reasoning and problem-solving
  - `deepseek-v4-flash` - Fast reasoning model with lower latency
  - `deepseek-v4-pro` - Latest reasoning model with thinking support

  ## JSON Output

  DeepSeek provides JSON Output to ensure the model outputs valid JSON strings.

  To use JSON output with `generate_object/3`:

      ReqLLM.generate_object("deepseek:deepseek-chat", "Extract information",
        schema: [
          name: [type: :string],
          age: [type: :integer],
          email: [type: :string]
        ]
      )

  This automatically sets `response_format` to `%{type: "json_object"}` and includes
  JSON formatting instructions in the system prompt.

  For manual control via `generate_text/3`:

      ReqLLM.generate_text("deepseek:deepseek-chat", "Return JSON only",
        provider_options: [response_format: %{type: "json_object"}],
        max_tokens: 1000
      )

  **Note:** When using JSON output, include "json" in your prompt and set reasonable
  `max_tokens` to prevent truncation.

  ## Thinking Mode

  DeepSeek models support thinking mode for improved reasoning.

  See [DeepSeek Thinking Mode Guide](https://api-docs.deepseek.com/guides/thinking_mode) for details.

  ### Options

  - `thinking: %{type: "enabled"}` - Enable thinking mode (default for reasoning models)
  - `thinking: %{type: "disabled"}` - Disable thinking mode

  ### Reasoning Effort

  The `reasoning_effort` option controls the depth of reasoning. For compatibility,
  `:low` and `:medium` are mapped to `"high"`, and `:xhigh` is mapped to `"max"`.

  Only `:high` and `:max` are the meaningful values sent to the API:

  - `:low` → mapped to `"high"`
  - `:medium` → mapped to `"high"`
  - `:high` → `"high"` (default for thinking mode)
  - `:xhigh` → mapped to `"max"`
  - `:max` → `"max"` (maximum effort for complex tasks)

  See https://platform.deepseek.com/docs for full model documentation.
  """

  use ReqLLM.Provider,
    id: :deepseek,
    default_base_url: "https://api.deepseek.com",
    default_env_key: "DEEPSEEK_API_KEY"

  use ReqLLM.Provider.Defaults

  import ReqLLM.Provider.Utils, only: [maybe_put: 3]

  @provider_schema [
    thinking: [
      type: :map,
      doc: """
      Thinking mode configuration. Set to %{type: "enabled"} to enable or %{type: "disabled"} to disable.
      Defaults to enabled for reasoning models. See https://api-docs.deepseek.com/guides/thinking_mode
      """
    ],
    response_format: [
      type: :map,
      doc: """
      Response format configuration. For JSON output, use: %{type: "json_object"}.
      When using json_object mode, include "json" in your prompt. See https://api-docs.deepseek.com/guides/json_mode
      """
    ]
  ]

  @impl ReqLLM.Provider
  def prepare_request(:object, model_spec, prompt, opts) do
    compiled_schema = Keyword.fetch!(opts, :compiled_schema)

    json_schema = ReqLLM.Schema.to_json(compiled_schema.schema)

    schema_name = Map.get(compiled_schema, :name, "output_schema")

    base_prompt = """
    Please output the result as a valid JSON object matching this schema:

    #{Jason.encode!(json_schema, pretty: true)}

    Output ONLY the JSON object, no additional text.
    """

    full_prompt = "#{base_prompt}\n\nUser request: #{prompt}"

    response_format = %{type: "json_object"}

    opts_with_format =
      opts
      |> Keyword.update(:provider_options, [response_format: response_format], fn existing ->
        Keyword.put(existing, :response_format, response_format)
      end)
      |> Keyword.put(:operation, :object)

    ReqLLM.Provider.Defaults.prepare_request(
      __MODULE__,
      :chat,
      model_spec,
      full_prompt,
      opts_with_format
    )
  end

  def prepare_request(operation, model_spec, input, opts) do
    ReqLLM.Provider.Defaults.prepare_request(__MODULE__, operation, model_spec, input, opts)
  end

  @impl ReqLLM.Provider
  def translate_options(_operation, _model, opts) do
    {reasoning_effort, opts} = Keyword.pop(opts, :reasoning_effort)

    opts =
      case reasoning_effort do
        :low -> Keyword.put(opts, :reasoning_effort, "high")
        :medium -> Keyword.put(opts, :reasoning_effort, "high")
        :high -> Keyword.put(opts, :reasoning_effort, "high")
        :max -> Keyword.put(opts, :reasoning_effort, "max")
        :xhigh -> Keyword.put(opts, :reasoning_effort, "max")
        nil -> opts
        other when is_binary(other) -> Keyword.put(opts, :reasoning_effort, other)
        other -> Keyword.put(opts, :reasoning_effort, to_string(other))
      end

    {opts, []}
  end

  @impl ReqLLM.Provider
  def build_body(request) do
    provider_opts = request.options[:provider_options] || []

    ReqLLM.Provider.Defaults.default_build_body(request)
    |> ensure_assistant_reasoning_content()
    |> maybe_put(:thinking, normalize_thinking(provider_opts[:thinking]))
    |> maybe_put(:reasoning_effort, request.options[:reasoning_effort])
    |> maybe_put(:response_format, provider_opts[:response_format])
  end

  defp ensure_assistant_reasoning_content(body) do
    case get_messages(body) do
      nil ->
        body

      messages when is_list(messages) ->
        key = messages_key(body)
        Map.put(body, key, Enum.map(messages, &add_reasoning_content_to_message/1))
    end
  end

  defp add_reasoning_content_to_message(msg) do
    if assistant_message?(msg) and not has_reasoning_content?(msg) do
      Map.put(msg, :reasoning_content, "")
    else
      msg
    end
  end

  defp get_messages(%{messages: msgs}), do: msgs
  defp get_messages(%{"messages" => msgs}), do: msgs
  defp get_messages(_), do: nil

  defp messages_key(%{messages: _}), do: :messages
  defp messages_key(%{"messages" => _}), do: "messages"

  defp assistant_message?(msg) when is_map(msg) do
    Map.get(msg, :role) == "assistant" or Map.get(msg, "role") == "assistant"
  end

  defp assistant_message?(_msg), do: false

  defp has_reasoning_content?(msg) when is_map(msg) do
    Map.has_key?(msg, :reasoning_content) or Map.has_key?(msg, "reasoning_content")
  end

  defp has_reasoning_content?(_msg), do: false

  defp normalize_thinking(nil), do: nil

  defp normalize_thinking(%{type: type} = thinking) when is_atom(type),
    do: %{thinking | type: to_string(type)}

  defp normalize_thinking(thinking) when is_map(thinking), do: thinking
end
