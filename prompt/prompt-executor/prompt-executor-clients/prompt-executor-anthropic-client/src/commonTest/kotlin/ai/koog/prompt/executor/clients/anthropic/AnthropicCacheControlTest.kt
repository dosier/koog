package ai.koog.prompt.executor.clients.anthropic

import ai.koog.prompt.executor.clients.anthropic.models.AnthropicCacheControl
import ai.koog.prompt.executor.clients.anthropic.models.AnthropicContent
import ai.koog.prompt.executor.clients.anthropic.models.AnthropicMessage
import ai.koog.prompt.executor.clients.anthropic.models.AnthropicMessageRequest
import ai.koog.prompt.executor.clients.anthropic.models.AnthropicMessageRequestSerializer
import ai.koog.prompt.executor.clients.anthropic.models.AnthropicResponse
import ai.koog.prompt.executor.clients.anthropic.models.AnthropicTool
import ai.koog.prompt.executor.clients.anthropic.models.AnthropicToolSchema
import ai.koog.prompt.executor.clients.anthropic.models.AnthropicUsage
import ai.koog.prompt.executor.clients.anthropic.models.SystemAnthropicMessage
import ai.koog.test.utils.runWithBothJsonConfigurations
import io.kotest.assertions.json.shouldContainJsonKey
import io.kotest.assertions.json.shouldEqualJson
import io.kotest.matchers.shouldBe
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlin.test.Test

class AnthropicCacheControlTest {

    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
        encodeDefaults = true
        explicitNulls = false
    }

    @Test
    fun `test serialization of system message with cache control`() =
        runWithBothJsonConfigurations("serialization of system message with cache control") { json ->
            val request = AnthropicMessageRequest(
                model = "claude-3-5-sonnet-20241022",
                messages = listOf(
                    AnthropicMessage.User(
                        content = listOf(AnthropicContent.Text("Hello"))
                    )
                ),
                maxTokens = 1000,
                system = listOf(
                    SystemAnthropicMessage(
                        text = "You are a helpful assistant.",
                        cacheControl = AnthropicCacheControl.Ephemeral
                    )
                )
            )

            val jsonString = json.encodeToString(AnthropicMessageRequestSerializer, request)

            jsonString shouldEqualJson
                // language=json
                """
            {
              "model": "claude-3-5-sonnet-20241022",
              "max_tokens": 1000,
              "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
              ],
              "stream": false,
              "system": [
                {
                  "type": "text",
                  "text": "You are a helpful assistant.",
                  "cache_control": {"type": "ephemeral"}
                }
              ]
            }
                """.trimIndent()
        }

    @Test
    fun `test serialization of tool with cache control`() =
        runWithBothJsonConfigurations("serialization of tool with cache control") { json ->
            val request = AnthropicMessageRequest(
                model = "claude-3-5-sonnet-20241022",
                messages = listOf(
                    AnthropicMessage.User(
                        content = listOf(AnthropicContent.Text("What's the weather?"))
                    )
                ),
                maxTokens = 1000,
                tools = listOf(
                    AnthropicTool(
                        name = "get_weather",
                        description = "Get the current weather for a location",
                        inputSchema = AnthropicToolSchema(
                            properties = JsonObject(emptyMap()),
                            required = emptyList()
                        ),
                        cacheControl = AnthropicCacheControl.Ephemeral
                    )
                )
            )

            val jsonString = json.encodeToString(AnthropicMessageRequestSerializer, request)

            jsonString shouldContainJsonKey "$.tools[0].cache_control"
        }

    @Test
    fun `test deserialization of response with cache metrics`() {
        val responseJson =
            // language=json
            """
            {
              "id": "msg_123",
              "type": "message",
              "role": "assistant",
              "content": [{"type": "text", "text": "Hello!"}],
              "model": "claude-3-5-sonnet-20241022",
              "stop_reason": "end_turn",
              "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 80,
                "cache_read_input_tokens": 20
              }
            }
            """.trimIndent()

        val response = json.decodeFromString<AnthropicResponse>(responseJson)

        response.usage shouldBe AnthropicUsage(
            inputTokens = 100,
            outputTokens = 50,
            cacheCreationInputTokens = 80,
            cacheReadInputTokens = 20
        )
    }

    @Test
    fun `test deserialization of response without cache metrics`() {
        val responseJson =
            // language=json
            """
            {
              "id": "msg_456",
              "type": "message",
              "role": "assistant",
              "content": [{"type": "text", "text": "Hi there!"}],
              "model": "claude-3-5-sonnet-20241022",
              "stop_reason": "end_turn",
              "usage": {
                "input_tokens": 50,
                "output_tokens": 25
              }
            }
            """.trimIndent()

        val response = json.decodeFromString<AnthropicResponse>(responseJson)

        response.usage shouldBe AnthropicUsage(
            inputTokens = 50,
            outputTokens = 25,
            cacheCreationInputTokens = null,
            cacheReadInputTokens = null
        )
    }

    @Test
    fun `test AnthropicCacheControl Ephemeral companion object`() {
        val cacheControl = AnthropicCacheControl.Ephemeral
        cacheControl.type shouldBe "ephemeral"
    }

    @Test
    fun `test AnthropicParams with cache control`() {
        val params = AnthropicParams(
            temperature = 0.7,
            maxTokens = 1000,
            cacheControl = ai.koog.prompt.params.LLMParams.CacheControl.Ephemeral
        )

        params.cacheControl shouldBe ai.koog.prompt.params.LLMParams.CacheControl.Ephemeral
    }
}
