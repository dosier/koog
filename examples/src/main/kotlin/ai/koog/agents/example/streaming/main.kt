package ai.koog.agents.example.streaming

import ai.koog.agents.core.agent.AIAgent
import ai.koog.agents.core.agent.config.AIAgentConfig
import ai.koog.agents.core.agent.entity.ToolSelectionStrategy
import ai.koog.agents.core.dsl.builder.forwardTo
import ai.koog.agents.core.dsl.builder.strategy
import ai.koog.agents.core.dsl.extension.nodeLLMRequestStreaming
import ai.koog.agents.core.tools.ToolRegistry
import ai.koog.agents.core.tools.annotations.InternalAgentToolsApi
import ai.koog.agents.example.ApiKeyService
import ai.koog.agents.features.eventHandler.feature.handleEvents
import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.executor.clients.anthropic.AnthropicClientSettings
import ai.koog.prompt.executor.clients.anthropic.AnthropicLLMClient
import ai.koog.prompt.executor.clients.anthropic.AnthropicModels
import ai.koog.prompt.executor.clients.openai.OpenAIModels
import ai.koog.prompt.executor.llms.SingleLLMPromptExecutor
import ai.koog.prompt.executor.llms.all.simpleAnthropicExecutor
import ai.koog.prompt.executor.llms.all.simpleOpenAIExecutor
import ai.koog.prompt.params.LLMParams
import ai.koog.prompt.streaming.StreamFrame
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.flow.onStart
import kotlinx.coroutines.flow.transform
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.encodeToString

@OptIn(InternalAgentToolsApi::class)
fun main() {
    val agent = AIAgent(
        promptExecutor = simpleAnthropicExecutor(ApiKeyService.anthropicApiKey),
        strategy = strategy<String, Flow<StreamFrame>>(
            name = "stream_api_strategy",
            toolSelectionStrategy = ToolSelectionStrategy.ALL,
        ) {
            val nodeCallLLM by nodeLLMRequestStreaming()
            edge(nodeStart forwardTo nodeCallLLM)
            edge(nodeCallLLM forwardTo nodeFinish)
        },
        agentConfig = AIAgentConfig(
            prompt = prompt(
                id = "flopiq-chat-system-prompt",
                params = LLMParams(
                    maxTokens = 1000,
                )
            ) {
                system("You talk back")
            },
            model = AnthropicModels.Sonnet_4,
            maxAgentIterations = 3
        ),
        installFeatures = {
            handleEvents {
                onStreamError {
                    it.error.printStackTrace()
                }
            }
        }
    )
    runBlocking {
        while(true) {
            val promptText = readln()
            agent.run(promptText).collect {
                println(it)
            }
        }
    }
}
