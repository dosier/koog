package ai.koog.prompt.executor.clients.anthropic

import ai.koog.http.client.KoogHttpClient
import ai.koog.http.client.ktor.fromKtorClient
import ai.koog.http.client.test.MockWebServer
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.streaming.StreamFrame
import io.github.oshai.kotlinlogging.KotlinLogging
import io.ktor.client.HttpClient
import io.ktor.client.engine.cio.CIO
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonPrimitive
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

/**
 * Regression: streaming [executeStreaming] must surface prompt-cache token splits on
 * [StreamFrame.End.metaInfo.metadata] (camelCase keys), matching the non-streaming path.
 */
class AnthropicStreamingCacheMetadataTest {

    private val sseJson = Json {
        ignoreUnknownKeys = true
        isLenient = true
        encodeDefaults = true
        explicitNulls = false
        namingStrategy = kotlinx.serialization.json.JsonNamingStrategy.SnakeCase
    }

    @Test
    fun `streaming message_delta carries cacheCreationInputTokens in metadata`(): Unit = runBlocking {
        val events = listOf(
            """{"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","content":[],"model":"claude-3-5-haiku-20241022","usage":{"input_tokens":10,"cache_creation_input_tokens":1234}}}""",
            """{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}""",
        )
        val mockServer = MockWebServer()
        mockServer.start(
            sseEndpoints = listOf(MockWebServer.SSEEndpointConfig("/v1/messages", events)),
        )
        try {
            val koogHttp = KoogHttpClient.fromKtorClient(
                clientName = "AnthropicStreamingCacheMetadataTest",
                logger = KotlinLogging.logger("AnthropicStreamingCacheMetadataTest"),
                baseClient = HttpClient(CIO.create()),
                baseUrl = mockServer.url(),
                requestTimeoutMillis = 60_000,
                connectTimeoutMillis = 10_000,
                socketTimeoutMillis = 120_000,
                json = sseJson,
                headers = mapOf(
                    "x-api-key" to "test-key",
                    "anthropic-version" to "2023-06-01",
                ),
            )

            val client = AnthropicLLMClient(
                settings = AnthropicClientSettings(
                    baseUrl = mockServer.url(),
                    messagesPath = "v1/messages",
                ),
                httpClient = koogHttp,
            )

            val prompt = Prompt.build("stream-cache-metadata") {
                user("Say hi.")
            }

            val frames =
                client.executeStreaming(prompt, AnthropicModels.Haiku_4_5, emptyList()).toList()
            val endFrames = frames.filterIsInstance<StreamFrame.End>()
            assertEquals(1, endFrames.size, "expected exactly one StreamFrame.End from message_delta")
            val md =
                assertNotNull(endFrames.single().metaInfo.metadata, "cache metadata missing on StreamFrame.End")
            assertEquals(1234, md["cacheCreationInputTokens"]?.jsonPrimitive?.intOrNull)
        } finally {
            mockServer.stop()
        }
    }
}
