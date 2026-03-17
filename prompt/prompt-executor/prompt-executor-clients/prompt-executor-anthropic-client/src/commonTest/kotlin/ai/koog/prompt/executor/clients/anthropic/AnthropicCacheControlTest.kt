package ai.koog.prompt.executor.clients.anthropic

import ai.koog.agents.core.tools.ToolDescriptor
import ai.koog.agents.core.tools.withCacheControl
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
import ai.koog.prompt.message.RequestMetaInfo
import ai.koog.prompt.params.LLMParams
import ai.koog.test.utils.runWithBothJsonConfigurations
import io.kotest.assertions.json.shouldContainJsonKey
import io.kotest.assertions.json.shouldEqualJson
import io.kotest.assertions.json.shouldNotContainJsonKey
import io.kotest.matchers.shouldBe
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNamingStrategy
import kotlinx.serialization.json.JsonObject
import kotlin.test.Test

class AnthropicCacheControlTest {

    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
        encodeDefaults = true
        explicitNulls = false
        namingStrategy = JsonNamingStrategy.SnakeCase
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

    @Test
    fun `test serialization of user message with cache control`() =
        runWithBothJsonConfigurations("serialization of user message with cache control") { json ->
            val request = AnthropicMessageRequest(
                model = "claude-3-5-sonnet-20241022",
                messages = listOf(
                    AnthropicMessage.User(
                        content = listOf(
                            AnthropicContent.Text(
                                text = "Hello, this is a long context that should be cached",
                                cacheControl = AnthropicCacheControl.Ephemeral
                            )
                        )
                    )
                ),
                maxTokens = 1000
            )

            val jsonString = json.encodeToString(AnthropicMessageRequestSerializer, request)

            jsonString shouldContainJsonKey "$.messages[0].content[0].cache_control"
            jsonString shouldEqualJson
                // language=json
                """
            {
              "model": "claude-3-5-sonnet-20241022",
              "max_tokens": 1000,
              "messages": [
                {
                  "role": "user",
                  "content": [
                    {
                      "type": "text",
                      "text": "Hello, this is a long context that should be cached",
                      "cache_control": {"type": "ephemeral"}
                    }
                  ]
                }
              ],
              "stream": false
            }
                """.trimIndent()
        }

    @Test
    fun `test serialization of assistant message with cache control`() =
        runWithBothJsonConfigurations("serialization of assistant message with cache control") { json ->
            val request = AnthropicMessageRequest(
                model = "claude-3-5-sonnet-20241022",
                messages = listOf(
                    AnthropicMessage.User(
                        content = listOf(AnthropicContent.Text("Hello"))
                    ),
                    AnthropicMessage.Assistant(
                        content = listOf(
                            AnthropicContent.Text(
                                text = "This is a previous response that should be cached",
                                cacheControl = AnthropicCacheControl.Ephemeral
                            )
                        )
                    ),
                    AnthropicMessage.User(
                        content = listOf(AnthropicContent.Text("Continue"))
                    )
                ),
                maxTokens = 1000
            )

            val jsonString = json.encodeToString(AnthropicMessageRequestSerializer, request)

            jsonString shouldContainJsonKey "$.messages[1].content[0].cache_control"
        }

    @Test
    fun `test serialization of tool result with cache control`() =
        runWithBothJsonConfigurations("serialization of tool result with cache control") { json ->
            val request = AnthropicMessageRequest(
                model = "claude-3-5-sonnet-20241022",
                messages = listOf(
                    AnthropicMessage.User(
                        content = listOf(
                            AnthropicContent.ToolResult(
                                toolUseId = "tool_123",
                                content = "Result from tool that should be cached",
                                cacheControl = AnthropicCacheControl.Ephemeral
                            )
                        )
                    )
                ),
                maxTokens = 1000
            )

            val jsonString = json.encodeToString(AnthropicMessageRequestSerializer, request)

            jsonString shouldContainJsonKey "$.messages[0].content[0].cache_control"
        }

    @Test
    fun `test serialization of tool use with cache control`() =
        runWithBothJsonConfigurations("serialization of tool use with cache control") { json ->
            val request = AnthropicMessageRequest(
                model = "claude-3-5-sonnet-20241022",
                messages = listOf(
                    AnthropicMessage.Assistant(
                        content = listOf(
                            AnthropicContent.ToolUse(
                                id = "tool_123",
                                name = "get_weather",
                                input = JsonObject(emptyMap()),
                                cacheControl = AnthropicCacheControl.Ephemeral
                            )
                        )
                    )
                ),
                maxTokens = 1000
            )

            val jsonString = json.encodeToString(AnthropicMessageRequestSerializer, request)

            jsonString shouldContainJsonKey "$.messages[0].content[0].cache_control"
        }

    @Test
    fun `test serialization without cache control when not specified`() =
        runWithBothJsonConfigurations("serialization without cache control when not specified") { json ->
            val request = AnthropicMessageRequest(
                model = "claude-3-5-sonnet-20241022",
                messages = listOf(
                    AnthropicMessage.User(
                        content = listOf(AnthropicContent.Text("Hello"))
                    )
                ),
                maxTokens = 1000
            )

            val jsonString = json.encodeToString(AnthropicMessageRequestSerializer, request)

            jsonString shouldNotContainJsonKey "$.messages[0].content[0].cache_control"
        }

    @Test
    fun `test RequestMetaInfo with cache control`() {
        val metaInfo = RequestMetaInfo.create(
            clock = kotlin.time.Clock.System,
            cacheControl = LLMParams.CacheControl.Ephemeral
        )

        metaInfo.cacheControl shouldBe LLMParams.CacheControl.Ephemeral
    }

    @Test
    fun `test RequestMetaInfo without cache control`() {
        val metaInfo = RequestMetaInfo.create(
            clock = kotlin.time.Clock.System
        )

        metaInfo.cacheControl shouldBe null
    }

    @Test
    fun `test multiple cache breakpoints in conversation`() =
        runWithBothJsonConfigurations("multiple cache breakpoints in conversation") { json ->
            val request = AnthropicMessageRequest(
                model = "claude-3-5-sonnet-20241022",
                messages = listOf(
                    AnthropicMessage.User(
                        content = listOf(
                            AnthropicContent.Text(
                                text = "First message with context",
                                cacheControl = AnthropicCacheControl.Ephemeral
                            )
                        )
                    ),
                    AnthropicMessage.Assistant(
                        content = listOf(AnthropicContent.Text("Response 1"))
                    ),
                    AnthropicMessage.User(
                        content = listOf(
                            AnthropicContent.Text(
                                text = "Second message with more context",
                                cacheControl = AnthropicCacheControl.Ephemeral
                            )
                        )
                    ),
                    AnthropicMessage.Assistant(
                        content = listOf(AnthropicContent.Text("Response 2"))
                    ),
                    AnthropicMessage.User(
                        content = listOf(AnthropicContent.Text("Final question"))
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

            // Verify cache control on system message
            jsonString shouldContainJsonKey "$.system[0].cache_control"
            // Verify cache control on first user message
            jsonString shouldContainJsonKey "$.messages[0].content[0].cache_control"
            // Verify cache control on third message (second user message)
            jsonString shouldContainJsonKey "$.messages[2].content[0].cache_control"
            // Verify no cache control on other messages
            jsonString shouldNotContainJsonKey "$.messages[1].content[0].cache_control"
            jsonString shouldNotContainJsonKey "$.messages[3].content[0].cache_control"
            jsonString shouldNotContainJsonKey "$.messages[4].content[0].cache_control"
        }

    // ===== Tool Descriptor Cache Control Tests =====

    @Test
    fun `test ToolDescriptor with cache control`() {
        val descriptor = ToolDescriptor(
            name = "get_weather",
            description = "Get current weather",
            cacheControl = LLMParams.CacheControl.Ephemeral
        )

        descriptor.cacheControl shouldBe LLMParams.CacheControl.Ephemeral
    }

    @Test
    fun `test ToolDescriptor copy with cache control`() {
        val original = ToolDescriptor(
            name = "get_weather",
            description = "Get current weather"
        )

        val withCache = original.copy(cacheControl = LLMParams.CacheControl.Ephemeral)

        original.cacheControl shouldBe null
        withCache.cacheControl shouldBe LLMParams.CacheControl.Ephemeral
        withCache.name shouldBe "get_weather"
    }

    @Test
    fun `test ToolDescriptor withCacheControl extension`() {
        val original = ToolDescriptor(
            name = "search_docs",
            description = "Search documentation"
        )

        val cached = original.withCacheControl(LLMParams.CacheControl.Ephemeral)

        cached.cacheControl shouldBe LLMParams.CacheControl.Ephemeral
        cached.name shouldBe original.name
        cached.description shouldBe original.description
    }

    @Test
    fun `test serialization of tool with cache control from descriptor`() =
        runWithBothJsonConfigurations("serialization of tool with cache control from descriptor") { json ->
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
                        name = "search_docs",
                        description = "Search documentation",
                        inputSchema = AnthropicToolSchema(
                            properties = JsonObject(emptyMap()),
                            required = emptyList()
                        ),
                        cacheControl = AnthropicCacheControl.Ephemeral
                    ),
                    AnthropicTool(
                        name = "get_weather",
                        description = "Get weather",
                        inputSchema = AnthropicToolSchema(
                            properties = JsonObject(emptyMap()),
                            required = emptyList()
                        )
                        // No cache control on second tool
                    )
                )
            )

            val jsonString = json.encodeToString(AnthropicMessageRequestSerializer, request)

            // First tool should have cache control
            jsonString shouldContainJsonKey "$.tools[0].cache_control"
            // Second tool should NOT have cache control
            jsonString shouldNotContainJsonKey "$.tools[1].cache_control"
        }

    @Test
    fun `test multiple tools with different cache breakpoints`() =
        runWithBothJsonConfigurations("multiple tools with different cache breakpoints") { json ->
            val request = AnthropicMessageRequest(
                model = "claude-3-5-sonnet-20241022",
                messages = listOf(
                    AnthropicMessage.User(
                        content = listOf(AnthropicContent.Text("Help me"))
                    )
                ),
                maxTokens = 1000,
                tools = listOf(
                    AnthropicTool(
                        name = "tool1",
                        description = "First tool",
                        inputSchema = AnthropicToolSchema(properties = JsonObject(emptyMap()), required = emptyList())
                    ),
                    AnthropicTool(
                        name = "tool2",
                        description = "Second tool - cache breakpoint",
                        inputSchema = AnthropicToolSchema(properties = JsonObject(emptyMap()), required = emptyList()),
                        cacheControl = AnthropicCacheControl.Ephemeral
                    ),
                    AnthropicTool(
                        name = "tool3",
                        description = "Third tool",
                        inputSchema = AnthropicToolSchema(properties = JsonObject(emptyMap()), required = emptyList())
                    ),
                    AnthropicTool(
                        name = "tool4",
                        description = "Fourth tool - another cache breakpoint",
                        inputSchema = AnthropicToolSchema(properties = JsonObject(emptyMap()), required = emptyList()),
                        cacheControl = AnthropicCacheControl.Ephemeral
                    )
                )
            )

            val jsonString = json.encodeToString(AnthropicMessageRequestSerializer, request)

            // First tool - no cache control
            jsonString shouldNotContainJsonKey "$.tools[0].cache_control"
            // Second tool - has cache control
            jsonString shouldContainJsonKey "$.tools[1].cache_control"
            // Third tool - no cache control
            jsonString shouldNotContainJsonKey "$.tools[2].cache_control"
            // Fourth tool - has cache control
            jsonString shouldContainJsonKey "$.tools[3].cache_control"
        }
}
