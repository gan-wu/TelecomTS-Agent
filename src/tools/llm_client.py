import os
import time
import logging
import random
from typing import Any
from openai import OpenAI
from openai import RateLimitError, APIConnectionError, InternalServerError

# Configure logging for troubleshooting.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

class LLMClient:
    """OpenAI-compatible LLM client used by local and online backends."""
    def __init__(
        self,
        api_key=None,
        base_url="https://api.deepseek.com",
        model_name="deepseek-chat",
        provider: str = "single",
        default_extra_body: dict[str, Any] | None = None,
    ):
        # Prefer explicit configuration and fall back to environment variables.
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            logging.warning("API key is missing. Set DEEPSEEK_API_KEY or pass api_key explicitly.")
            raise ValueError("API Key is missing.")
            
        self.base_url = base_url
        self.model_name = model_name
        self.provider = provider
        self.default_extra_body = default_extra_body or {}
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        
    def query(
        self,
        messages,
        temperature=0.0,
        max_tokens=256,
        max_retries=5,
        task: str | None = None,
        allow_reasoning_fallback: bool = False,
        route_signals: dict[str, Any] | None = None,
    ):
        """
        Run one chat completion request with exponential backoff and jitter.
        """
        retries = 0
        base_delay = 1.0  # Base delay for exponential backoff.
        
        while retries <= max_retries:
            try:
                # Use low temperature for deterministic answer extraction when requested.
                request_kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if self.default_extra_body:
                    request_kwargs["extra_body"] = self.default_extra_body

                response = self.client.chat.completions.create(**request_kwargs)
                message = response.choices[0].message
                usage = getattr(response, "usage", None)
                self.last_usage = usage.model_dump() if hasattr(usage, "model_dump") else usage
                content = message.content or ""
                if content.strip():
                    return content

                reasoning_content = getattr(message, "reasoning_content", None) or ""
                if reasoning_content.strip():
                    if not allow_reasoning_fallback:
                        logging.debug(
                            "[%s/%s] Empty content with reasoning_content; triggering backend fallback for task=%s",
                            self.provider,
                            self.model_name,
                            task,
                        )
                        return None
                    logging.debug(
                        "[%s/%s] Empty content; returning reasoning_content fallback for task=%s",
                        self.provider,
                        self.model_name,
                        task,
                    )
                    return reasoning_content
                return None
                
            except RateLimitError as e:
                # Rate limits are often caused by high concurrency or provider quota constraints.
                logging.warning(f"[RateLimit] {str(e)}")
            except (APIConnectionError, InternalServerError) as e:
                # Retry transient network or provider-side failures.
                logging.warning(f"[Network/Server] {str(e)}")
            except Exception as e:
                # Raise non-transient errors immediately.
                logging.error(f"[Fatal] Non-retryable local error: {str(e)}")
                raise e
            
            # Retry with exponential backoff.
            retries += 1
            if retries > max_retries:
                logging.error(f"Maximum retries ({max_retries}) exhausted; returning empty response.")
                return None
                
            # Exponential backoff with jitter reduces synchronized retry spikes.
            delay = min(base_delay * (2 ** (retries - 1)), 60.0) 
            jitter = random.uniform(0, 0.5)
            wait_time = delay + jitter
            
            logging.info(f"Retry {retries}/{max_retries} after {wait_time:.2f}s backoff.")
            time.sleep(wait_time)
