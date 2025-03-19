import copy
import json
import logging
import os
from functools import cached_property
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import asyncio

from lm_eval.api.registry import register_model
from lm_eval.models.api_models import TemplateAPI, JsonChatStr
from lm_eval.models.utils import handle_stop_sequences
import requests
from openai import OpenAI
from openai.types.chat import ChatCompletion

try:
    from tenacity import RetryError
    from aiohttp import ClientSession, ClientTimeout, TCPConnector
except ModuleNotFoundError:
    pass


eval_logger = logging.getLogger(__name__)


@register_model("local-completions")
class LocalCompletionsAPI(TemplateAPI):
    def __init__(
        self,
        base_url=None,
        tokenizer_backend="huggingface",
        **kwargs,
    ):
        super().__init__(
            base_url=base_url, tokenizer_backend=tokenizer_backend, **kwargs
        )

    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        generate=False,
        gen_kwargs: Optional[dict] = None,
        seed: int = 1234,
        eos=None,
        **kwargs,
    ) -> dict:
        if generate:
            gen_kwargs.pop("do_sample", False)
            if "max_tokens" in gen_kwargs:
                max_tokens = gen_kwargs.pop("max_tokens")
            else:
                max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
            temperature = gen_kwargs.pop("temperature", 0)
            stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
            return {
                "prompt": messages,
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop,
                "seed": seed,
                **gen_kwargs,
            }
        else:
            return {
                "model": self.model,
                "prompt": messages,
                "temperature": 0,
                "max_tokens": 1,
                "logprobs": 1,
                "seed": seed,
                "echo": True,
            }

    @staticmethod
    def parse_logprobs(
        outputs: Union[Dict, List[Dict]],
        tokens: List[List[int]] = None,
        ctxlens: List[int] = None,
        **kwargs,
    ) -> List[Tuple[float, bool]]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            for choice, ctxlen in zip(
                sorted(out["choices"], key=itemgetter("index")), ctxlens
            ):
                assert ctxlen > 0, "Context length must be greater than 0"
                logprobs = sum(choice["logprobs"]["token_logprobs"][ctxlen:-1])
                tokens_logprobs = choice["logprobs"]["token_logprobs"][ctxlen:-1]
                top_logprobs = choice["logprobs"]["top_logprobs"][ctxlen:-1]
                is_greedy = True
                for tok, top in zip(tokens_logprobs, top_logprobs):
                    if tok != max(top.values()):
                        is_greedy = False
                        break
                res.append((logprobs, is_greedy))
        return res

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            tmp = [None] * len(out["choices"])
            for choices in out["choices"]:
                tmp[choices["index"]] = choices["text"]
            res = res + tmp
        return res

    @property
    def api_key(self):
        return os.environ.get("OPENAI_API_KEY", "")


@register_model("local-chat-completions")
class LocalChatCompletion(LocalCompletionsAPI):
    def __init__(
        self,
        base_url=None,
        tokenizer_backend=None,
        tokenized_requests=False,
        **kwargs,
    ):
        eval_logger.warning(
            "chat-completions endpoint requires the `--apply_chat_template` flag."
        )
        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )
        if self._batch_size > 1:
            eval_logger.warning(
                "Chat completions does not support batching. Defaulting to batch size 1."
            )
            self._batch_size = 1

    def _create_payload(
        self,
        messages: List[Dict],
        generate=False,
        gen_kwargs: dict = None,
        seed=1234,
        eos=None,
        **kwargs,
    ) -> dict:
        assert type(messages) is not str, (
            "chat-completions require the --apply_chat_template flag."
        )
        gen_kwargs.pop("do_sample", False)
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop]
        return {
            "messages": messages,
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop[:4],
            "seed": seed,
            **gen_kwargs,
        }

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            tmp = [None] * len(out["choices"])
            for choices in out["choices"]:
                tmp[choices["index"]] = choices["message"]["content"]
            res = res + tmp
        return res

    def tok_encode(
        self,
        string: Union[str, Any],
        left_truncate_len=None,
        add_special_tokens=None,
        **kwargs,
    ) -> Union[List[str], List[int], Any]:
        return string

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "Loglikelihood is not supported for chat completions. Consider using the completions API instead."
        )
    

@register_model("local-reasoning-completions")
class LocalReasoningCompletion(LocalChatCompletion):
    def __init__(
        self,
        base_url=None,
        tokenizer_backend=None,
        tokenized_requests=False,
        **kwargs,
    ):
        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )
    
    def _create_payload(
        self,
        messages: List[Dict],
        generate=False,
        gen_kwargs: dict = None,
        seed=1234,
        eos=None,
        **kwargs,
    ) -> dict:
        """Override to add stream=True for streaming support"""
        payload = super()._create_payload(
            messages=messages,
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=seed,
            eos=eos,
            **kwargs,
        )
        payload["stream"] = True  # Enable streaming
        payload["stream_options"] = {"include_usage": True}  # Include usage stats
        return payload
    
    def _process_chunk(self, chunk, state):
        """Process a single chunk from the streaming response and update state.
        
        Args:
            chunk: The JSON chunk from the streaming response
            state: A dictionary containing the current state of the response processing
            
        Returns:
            None (updates state in-place)
        """
        # Update metadata if available
        state["created"] = chunk.get('created', state["created"])
        state["system_fingerprint"] = chunk.get('system_fingerprint', state["system_fingerprint"])
        state["model"] = chunk.get('model', state["model"])
        state["id"] = chunk.get('id', state["id"])
        
        # Get usage information from the last chunk
        if chunk.get('usage') is not None:
            state["usage"] = chunk['usage']
            
        # Process content from choices
        if chunk.get('choices') and len(chunk['choices']) > 0:
            delta = chunk['choices'][0].get('delta', {})
            
            # Collect reasoning content
            if delta.get('reasoning_content') is not None:
                state["reasoning_content"] += delta.get('reasoning_content', '')
            
            # Collect regular content
            if delta.get('content') is not None:
                state["content"] += delta.get('content', '')
                
            # Get finish_reason if present
            finish_reason = chunk['choices'][0].get('finish_reason')
            if finish_reason and not state["choices"]:
                # Initialize choices with finish_reason when we first see it
                state["choices"] = [{
                    "index": chunk['choices'][0].get('index', 0),
                    "finish_reason": finish_reason,
                    "logprobs": chunk['choices'][0].get('logprobs')
                }]
    
    def _build_final_response(self, state):
        """Build the final response object from the state.
        
        Args:
            state: A dictionary containing the current state of the response processing
            
        Returns:
            dict: The final response object in the format expected by the OpenAI API
        """
        # Build the final response object
        combined_content = ""
        if state["reasoning_content"]:
            combined_content = f"<think>{state['reasoning_content']}</think>"
        combined_content += state["content"]
        
        # Create a choice with the combined content
        if not state["choices"]:
            state["choices"] = [{
                "index": 0,
                "finish_reason": "stop",
                "logprobs": None
            }]
        
        # Add message with the combined content to the first choice
        state["choices"][0]["message"] = {
            "content": combined_content,
            "reasoning_content": state["reasoning_content"],
            "role": "assistant"
        }
        
        # Construct the final response object
        return {
            "choices": state["choices"],
            "object": "chat.completion",
            "usage": state["usage"] or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "created": state["created"] or int(time.time()),
            "system_fingerprint": state["system_fingerprint"],
            "model": state["model"],
            "id": state["id"] or f"chatcmpl-{int(time.time())}"
        }
    
    def _process_streaming_response(self, response):
        """Process a streaming response and build the final response object.
        
        This encapsulates all the state tracking in one function, making the code
        cleaner and easier to maintain.
        """
        # Initialize state
        state = {
            "reasoning_content": "",
            "content": "",
            "choices": [],
            "usage": None,
            "created": None,
            "system_fingerprint": None,
            "model": self.model,
            "id": None
        }
        
        try:
            # Process each line in the response
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        line = line[6:]  # Remove 'data: ' prefix
                        if line == '[DONE]':
                            break
                            
                        try:
                            chunk = json.loads(line)
                            self._process_chunk(chunk, state)
                        except json.JSONDecodeError:
                            continue
            
            # Build and return the final response object
            return self._build_final_response(state)
        except Exception as e:
            eval_logger.error(f"Error in _process_streaming_response: {e}")
    
    async def _process_streaming_response_async(self, response):
        """Process a streaming response asynchronously and build the final response object."""
        # Initialize state
        state = {
            "reasoning_content": "",
            "content": "",
            "choices": [],
            "usage": None,
            "created": None,
            "system_fingerprint": None,
            "model": self.model,
            "id": None
        }
        
        try:
            # Process each chunk in the response
            async for chunk_bytes in response.content:
                if chunk_bytes:
                    chunk_str = chunk_bytes.decode('utf-8')
                    for line in chunk_str.splitlines():
                        if line.startswith('data: '):
                            line = line[6:]  # Remove 'data: ' prefix
                            if line == '[DONE]':
                                continue
                                
                            try:
                                chunk = json.loads(line)
                                self._process_chunk(chunk, state)
                            except json.JSONDecodeError:
                                continue
            
            # Build and return the final response object
            return self._build_final_response(state)
        except Exception as e:
            eval_logger.error(f"Error in _process_streaming_response_async: {e}")
            return None
    
    def model_call(
        self,
        messages: Union[List[List[int]], List[str], List[JsonChatStr]],
        *,
        generate: bool = True,
        gen_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> Optional[dict]:
        # !!! Copy: shared dict for each request, need new object !!!
        gen_kwargs = copy.deepcopy(gen_kwargs)
        try:
            payload = self._create_payload(
                self.create_message(messages),
                generate=generate,
                gen_kwargs=gen_kwargs,
                seed=self._seed,
                eos=self.eos_string,
                **kwargs,
            )
            
            # Make the streaming API request with an explicit timeout
            response = requests.post(
                self.base_url,
                json=payload,
                headers=self.header,
                verify=self.verify_certificate,
                stream=True
            )
            
            if not response.ok:
                eval_logger.warning(
                    f"API request failed with error message: {response.text}. Retrying..."
                )
            response.raise_for_status()
            
            # Process the streaming response
            result = self._process_streaming_response(response)
            return result
            
        except RetryError:
            eval_logger.error(
                "API request failed after multiple retries. Please check the API status."
            )
            return None

    async def amodel_call(
        self,
        session: ClientSession,
        messages: Union[List[List[int]], List[str], List[JsonChatStr]],
        *,
        generate: bool = True,
        cache_keys: list = None,
        ctxlens: Optional[List[int]] = None,
        gen_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> Union[List[str], List[Tuple[float, bool]], None]:
        """Async version of model_call that handles streaming responses"""
        # !!! Copy: shared dict for each request, need new object !!!
        gen_kwargs = copy.deepcopy(gen_kwargs)
        
        try:
            payload = self._create_payload(
                self.create_message(messages),
                generate=generate,
                gen_kwargs=gen_kwargs,
                seed=self._seed,
                eos=self.eos_string,
                **kwargs,
            )
            
            cache_method = "generate_until" if generate else "loglikelihood"
            
            async with session.post(
                self.base_url,
                json=payload,
                headers=self.header,
            ) as response:
                if not response.ok:
                    error_text = await response.text()
                    eval_logger.warning(
                        f"API request failed with error message: {error_text}. Retrying..."
                    )
                
                # Raising exception will retry the request
                response.raise_for_status()
                
                # Process the streaming response
                outputs = await self._process_streaming_response_async(response)
                
            # Parse the outputs based on whether we're generating or getting logprobs
            answers = (
                self.parse_generations(
                    outputs=outputs,
                )
                if generate
                else self.parse_logprobs(
                    outputs=outputs,
                    tokens=messages,
                    ctxlens=ctxlens,
                )
            )
            
            # Cache results if requested
            if cache_keys:
                for res, cache in zip(answers, cache_keys):
                    self.cache_hook.add_partial(cache_method, cache, res)
            
            return answers
            
        except RetryError:
            eval_logger.error(
                "API request failed after multiple retries. Please check the API status."
            )
            return None


@register_model(
    "openai-completions",
)
class OpenAICompletionsAPI(LocalCompletionsAPI):
    def __init__(
        self,
        base_url="https://api.openai.com/v1/completions",
        tokenizer_backend="tiktoken",
        **kwargs,
    ):
        super().__init__(
            base_url=base_url, tokenizer_backend=tokenizer_backend, **kwargs
        )

    @cached_property
    def api_key(self):
        """Override this property to return the API key for the API request."""
        key = os.environ.get("OPENAI_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `OPENAI_API_KEY` environment variable."
            )
        return key

    def loglikelihood(self, requests, **kwargs):
        assert self.model in [
            "babbage-002",
            "davinci-002",
        ], (
            f"Prompt loglikelihoods are only supported by OpenAI's API for {['babbage-002', 'davinci-002']}."
        )
        return super().loglikelihood(requests, **kwargs)

    def chat_template(self, chat_template: Union[bool, str] = False) -> Optional[str]:
        return ""


@register_model("openai-chat-completions")
class OpenAIChatCompletion(LocalChatCompletion):
    def __init__(
        self,
        base_url="https://api.openai.com/v1/chat/completions",
        tokenizer_backend=None,
        tokenized_requests=False,
        **kwargs,
    ):
        if "o1" in kwargs.get("model", ""):
            eval_logger.warning(
                "o1 models do not support `stop` and only support temperature=1"
            )
        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )

    @cached_property
    def api_key(self):
        """Override this property to return the API key for the API request."""
        key = os.environ.get("OPENAI_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `OPENAI_API_KEY` environment variable."
            )
        return key

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "Loglikelihood (and therefore `multiple_choice`-type tasks) is not supported for chat completions as OpenAI does not provide prompt logprobs. See https://github.com/EleutherAI/lm-evaluation-harness/issues/942#issuecomment-1777836312 or https://github.com/EleutherAI/lm-evaluation-harness/issues/1196 for more background on this limitation."
        )

    def _create_payload(
        self,
        messages: List[Dict],
        generate=False,
        gen_kwargs: dict = None,
        seed=1234,
        eos="<|endoftext|>",
        **kwargs,
    ) -> dict:
        assert type(messages) is not str, (
            "chat-completions require the --apply_chat_template flag."
        )
        gen_kwargs.pop("do_sample", False)
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", ["<|endoftext|>"]), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop]
        output = {
            "messages": messages,
            "model": self.model,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop[:4],
            "seed": seed,
            **gen_kwargs,
        }
        if "o1" in self.model:
            output.pop("stop")
            output["temperature"] = 1
        elif "o3" in self.model:
            output.pop("temperature")
        return output


@register_model("openai-reasoning-completions")
class OpenAIReasoningCompletion(OpenAIChatCompletion):
    def __init__(
        self,
        base_url="https://api.openai.com/v1/chat/completions",
        tokenizer_backend=None,
        tokenized_requests=False,
        **kwargs,
    ):
        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
    
    def model_call(
        self,
        messages: Union[List[List[int]], List[str], List[JsonChatStr]],
        *,
        generate: bool = True,
        gen_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> Optional[dict]:
        # !!! Copy: shared dict for each request, need new object !!!
        gen_kwargs = copy.deepcopy(gen_kwargs)
        try:
            payload = self._create_payload(
                self.create_message(messages),
                generate=generate,
                gen_kwargs=gen_kwargs,
                seed=self._seed,
                eos=self.eos_string,
                **kwargs,
            )
            
            # Initialize response structure
            full_response = {"choices": []}
            
            # Create and send streaming request
            response_stream = self.client.chat.completions.create(**payload, stream=True)
            
            # Track each choice_index separately
            choice_data = {}
            
            for chunk in response_stream:
                for choice in chunk.choices:
                    choice_index = choice.index
                    
                    # Initialize data structure for new choices
                    if choice_index not in choice_data:
                        choice_data[choice_index] = {
                            "reasoning_content": "",
                            "content": "",
                            "finish_reason": None
                        }
                    
                    # Record finish_reason if present
                    if hasattr(choice, "finish_reason") and choice.finish_reason is not None:
                        choice_data[choice_index]["finish_reason"] = choice.finish_reason
                    
                    # Process delta content
                    if hasattr(choice, "delta"):
                        if hasattr(choice.delta, "reasoning_content") and choice.delta.reasoning_content is not None:
                            choice_data[choice_index]["reasoning_content"] += choice.delta.reasoning_content
                        
                        if hasattr(choice.delta, "content") and choice.delta.content is not None:
                            choice_data[choice_index]["content"] += choice.delta.content
            
            # Build complete response with all choices
            for idx, data in choice_data.items():
                full_response["choices"].append({
                    "index": idx,
                    "message": {
                        "content": data["content"],
                        "reasoning_content": data["reasoning_content"],
                        "role": "assistant"
                    },
                    "finish_reason": data["finish_reason"] or "stop"
                })
            
            return full_response
        except Exception as e:
            eval_logger.error(f"Error in model_call: {e}")
            return None
    
    async def amodel_call(
        self,
        session: ClientSession,
        messages: Union[List[List[int]], List[str], List[JsonChatStr]],
        *,
        generate: bool = True,
        cache_keys: list = None,
        ctxlens: Optional[List[int]] = None,
        gen_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> Union[List[str], List[Tuple[float, bool]], None]:
        """Async version of model_call that handles streaming responses"""
        # Copy shared dict for each request
        gen_kwargs = copy.deepcopy(gen_kwargs)
        
        try:
            payload = self._create_payload(
                self.create_message(messages),
                generate=generate,
                gen_kwargs=gen_kwargs,
                seed=self._seed,
                eos=self.eos_string,
                **kwargs,
            )
            
            cache_method = "generate_until" if generate else "loglikelihood"
            
            # Initialize response structure
            full_response = {"choices": []}
            
            # Create and send streaming request using the OpenAI client
            # Run the client.chat.completions.create method in a separate thread
            # to avoid blocking the event loop
            response_stream = await asyncio.to_thread(
                self.client.chat.completions.create,
                **payload,
                stream=True
            )
            
            # Track each choice_index separately
            choice_data = {}
            
            # Process the stream - MODIFIED: use regular for loop instead of async for
            # since Stream object doesn't support async iteration
            for chunk in response_stream:
                for choice in chunk.choices:
                    choice_index = choice.index
                    
                    # Initialize data structure for new choices
                    if choice_index not in choice_data:
                        choice_data[choice_index] = {
                            "reasoning_content": "",
                            "content": "",
                            "finish_reason": None
                        }
                    
                    # Record finish_reason if present
                    if hasattr(choice, "finish_reason") and choice.finish_reason is not None:
                        choice_data[choice_index]["finish_reason"] = choice.finish_reason
                    
                    # Process delta content
                    if hasattr(choice, "delta"):
                        if hasattr(choice.delta, "reasoning_content") and choice.delta.reasoning_content is not None:
                            choice_data[choice_index]["reasoning_content"] += choice.delta.reasoning_content
                        
                        if hasattr(choice.delta, "content") and choice.delta.content is not None:
                            choice_data[choice_index]["content"] += choice.delta.content
            
            # Build complete response with all choices
            for idx, data in choice_data.items():
                full_response["choices"].append({
                    "index": idx,
                    "message": {
                        "content": data["content"],
                        "reasoning_content": data["reasoning_content"],
                        "role": "assistant"
                    },
                    "finish_reason": data["finish_reason"] or "stop"
                })
            
            # Parse the outputs
            answers = (
                self.parse_generations(outputs=full_response)
                if generate
                else self.parse_logprobs(
                    outputs=full_response,
                    tokens=messages,
                    ctxlens=ctxlens,
                )
            )
            
            # Cache results if requested
            if cache_keys:
                for res, cache in zip(answers, cache_keys):
                    self.cache_hook.add_partial(cache_method, cache, res)
            
            return answers
            
        except Exception as e:
            eval_logger.error(f"Error in amodel_call: {e}")
            return None
        
    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            tmp = [None] * len(out["choices"])
            for choice in out["choices"]:
                index = choice["index"]
                message = choice["message"]
                if "reasoning_content" in message and message["reasoning_content"]:
                    tmp[index] = "<think>\n" + message["reasoning_content"] + "\n</think>\n" + message["content"]
                else:
                    tmp[index] = message["content"]
            res = res + tmp
        return res