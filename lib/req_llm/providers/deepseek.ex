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

  ## Models

  DeepSeek offers several models including:

  - `deepseek-chat` - General purpose conversational model
  - `deepseek-reasoner` - Reasoning and problem-solving

  See https://platform.deepseek.com/docs for full model documentation.
  """

  use ReqLLM.Provider,
    id: :deepseek,
    default_base_url: "https://api.deepseek.com",
    default_env_key: "DEEPSEEK_API_KEY"

  use ReqLLM.Provider.Defaults

  @provider_schema []

  @impl ReqLLM.Provider
  def build_body(request) do
    body = ReqLLM.Provider.Defaults.default_build_body(request)

    # Inject reasoning_content for assistant messages
    # For Deepseek V4, `reasoning_content` must always exist, even if empty
    messages =
      case Map.get(body, :messages) do
        nil ->
          nil

        msgs when is_list(msgs) ->
          # Add an empty `reasoning_content` field to all assistant messages if not already present
          Enum.map(msgs, fn msg ->
            if Map.get(msg, :role) == "assistant" or Map.get(msg, "role") == "assistant" do
              Map.put_new(msg, :reasoning_content, "")
            else
              msg
            end
          end)

        other ->
          other
      end

    if messages do
      Map.put(body, :messages, messages)
    else
      body
    end
  end
  # @impl ReqLLM.Provider
  # def decode_stream_event(%{data: data} = event, model) when is_map(data) do
  #   standard_chunks = ReqLLM.Provider.Defaults.default_decode_stream_event(event, model)

  #   case Map.get(data, "choices") do
  #     choices when is_list(choices) ->
  #       extra_chunks =
  #         Enum.flat_map(choices, fn choice ->
  #           delta = Map.get(choice, "delta", %{})
  #           has_content = Map.has_key?(delta, "content") and delta["content"] != ""

  #           if has_content do
  #             reasoning = Map.get(delta, "reasoning_content") || Map.get(delta, "reasoning")
  #             if is_binary(reasoning) and reasoning != "" do
  #               [ReqLLM.StreamChunk.thinking(reasoning)]
  #             else
  #               []
  #             end
  #           else
  #             []
  #           end
  #         end)

  #       extra_chunks ++ standard_chunks

  #     _ ->
  #       standard_chunks
  #   end
  # end

  # def decode_stream_event(event, model) do
  #   ReqLLM.Provider.Defaults.default_decode_stream_event(event, model)
  # end
end
