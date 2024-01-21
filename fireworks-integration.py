from dataclasses import Field
from typing import List, Optional, Type, Any
from cat.mad_hatter.decorators import tool, hook, plugin
from pydantic import BaseModel, ConfigDict, SecretStr
from cat.factory.llm import LLMSettings
from langchain_community.chat_models.fireworks import ChatFireworks
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import BaseMessage
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import ChatResult

# langchain is so facking cool!
# Monkey patching for enable streaming tokeks when using LLMchains
def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = self.stream
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        message_dicts = self._create_message_dicts(messages)

        params = {
            "model": self.model,
            "messages": message_dicts,
            **self.model_kwargs,
            **kwargs,
        }
        response = completion_with_retry(  # noqa: F821
            self,
            self.use_retry,
            run_manager=run_manager,
            stop=stop,
            **params,
        )
        return self._create_chat_result(response)
    
llm_fireworks = ChatFireworks
llm_fireworks._generate = _generate

class FireWorksAIConfig(LLMSettings):
    """The configuration for the FireworksAI plugin."""
    fireworks_api_key: Optional[SecretStr]
    model: str = "accounts/fireworks/models/llama-v2-7b-chat"
    temperature: int = 0.7
    max_tokens: int = 512
    top_p: float = 1
    top_k: int = 50
    stream: bool = True
    
    _pyclass: Type = llm_fireworks

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "FireworksAI",
            "description": "Configuration for FireworksAI",
            "link": "https://app.fireworks.ai/",
        }
    )


@hook
def factory_allowed_llms(allowed, cat) -> List:
    allowed.append(FireWorksAIConfig)
    return allowed
