"""
LLM Client Module
=================
Unified client for LLM API calls with retry logic, structured output parsing,
and graceful error handling. Foundation for all AI agents in the system.
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
    LOCAL = "local"  # Generic OpenAI-compatible local endpoint


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    provider: LLMProvider = LLMProvider.OLLAMA
    model: str = "llama3.1"
    api_key: Optional[str] = None
    api_key_env: str = "ANTHROPIC_API_KEY"
    base_url: Optional[str] = None  # For local LLMs (e.g., http://localhost:11434)
    max_retries: int = 3
    timeout_seconds: int = 120  # Local LLMs can be slower
    retry_delay: float = 1.0
    max_tokens: int = 4096
    temperature: float = 0.0  # Deterministic for consistency

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'LLMConfig':
        """Create config from dictionary."""
        llm_config = config.get('llm', {})
        provider_str = llm_config.get('provider', 'ollama')

        return cls(
            provider=LLMProvider(provider_str),
            model=llm_config.get('model', 'llama3.1'),
            api_key=llm_config.get('api_key'),
            api_key_env=llm_config.get('api_key_env', 'ANTHROPIC_API_KEY'),
            base_url=llm_config.get('base_url'),
            max_retries=llm_config.get('max_retries', 3),
            timeout_seconds=llm_config.get('timeout_seconds', 120),
            retry_delay=llm_config.get('retry_delay', 1.0),
            max_tokens=llm_config.get('max_tokens', 4096),
            temperature=llm_config.get('temperature', 0.0),
        )


@dataclass
class LLMResponse:
    """Structured response from LLM."""
    content: str
    parsed_json: Optional[Dict[str, Any]] = None
    model: str = ""
    usage: Optional[Dict[str, int]] = None
    success: bool = True
    error: Optional[str] = None

    @property
    def json(self) -> Dict[str, Any]:
        """Return parsed JSON or empty dict."""
        return self.parsed_json or {}


class LLMClient:
    """
    Unified LLM client with retry logic and structured output parsing.

    Usage:
        client = LLMClient.from_config(config)
        response = client.call(
            prompt="Analyze this strategy...",
            system="You are a quant trading expert.",
            response_format="json"
        )
        if response.success:
            decision = response.json.get('decision')
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
        self._initialized = False

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'LLMClient':
        """Create client from application config dictionary."""
        llm_config = LLMConfig.from_dict(config)
        return cls(llm_config)

    def _initialize(self) -> bool:
        """Lazy initialization of the API client."""
        if self._initialized:
            return self._client is not None

        self._initialized = True

        try:
            # Local LLM providers (no API key needed)
            if self.config.provider in (LLMProvider.OLLAMA, LLMProvider.LMSTUDIO, LLMProvider.LOCAL):
                return self._initialize_local()

            # Cloud providers (API key required)
            api_key = self.config.api_key or os.environ.get(self.config.api_key_env)

            if not api_key:
                logger.warning(
                    f"No API key found for {self.config.provider.value}. "
                    f"Set {self.config.api_key_env} environment variable or use a local LLM."
                )
                return False

            if self.config.provider == LLMProvider.ANTHROPIC:
                import anthropic
                self._client = anthropic.Anthropic(api_key=api_key)
            elif self.config.provider == LLMProvider.OPENAI:
                import openai
                self._client = openai.OpenAI(api_key=api_key)
            return True

        except ImportError as e:
            logger.error(f"Failed to import {self.config.provider.value} library: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            return False

    def _initialize_local(self) -> bool:
        """Initialize local LLM client (Ollama, LM Studio, etc.)."""
        try:
            import requests
            self._requests = requests

            # Determine base URL
            if self.config.base_url:
                self._base_url = self.config.base_url.rstrip('/')
            elif self.config.provider == LLMProvider.OLLAMA:
                self._base_url = "http://localhost:11434"
            elif self.config.provider == LLMProvider.LMSTUDIO:
                self._base_url = "http://localhost:1234"
            else:
                self._base_url = "http://localhost:8000"

            # Test connection
            try:
                if self.config.provider == LLMProvider.OLLAMA:
                    # Ollama uses /api/tags to list models
                    resp = requests.get(f"{self._base_url}/api/tags", timeout=5)
                else:
                    # OpenAI-compatible endpoints use /v1/models
                    resp = requests.get(f"{self._base_url}/v1/models", timeout=5)

                if resp.status_code == 200:
                    logger.info(f"Connected to local LLM at {self._base_url}")
                    self._client = "local"  # Mark as initialized
                    return True
                else:
                    logger.warning(f"Local LLM server responded with status {resp.status_code}")
            except requests.exceptions.ConnectionError:
                logger.warning(f"Could not connect to local LLM at {self._base_url}")
            except Exception as e:
                logger.warning(f"Error testing local LLM connection: {e}")

            return False

        except ImportError:
            logger.error("requests library not available for local LLM")
            return False

    def call(
        self,
        prompt: str,
        system: Optional[str] = None,
        response_format: str = "text",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """
        Make an LLM API call with retry logic.

        Args:
            prompt: The user prompt/question
            system: System message for context setting
            response_format: "text" or "json" (will parse JSON from response)
            max_tokens: Override default max tokens
            temperature: Override default temperature

        Returns:
            LLMResponse with content and optionally parsed JSON
        """
        if not self._initialize():
            return LLMResponse(
                content="",
                success=False,
                error="LLM client not initialized. Check API key configuration."
            )

        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature if temperature is not None else self.config.temperature

        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                if self.config.provider == LLMProvider.ANTHROPIC:
                    response = self._call_anthropic(prompt, system, max_tokens, temperature)
                elif self.config.provider in (LLMProvider.OLLAMA, LLMProvider.LMSTUDIO, LLMProvider.LOCAL):
                    response = self._call_local(prompt, system, max_tokens, temperature)
                else:
                    response = self._call_openai(prompt, system, max_tokens, temperature)

                # Parse JSON if requested
                if response_format == "json" and response.success:
                    response.parsed_json = self._parse_json(response.content)
                    if response.parsed_json is None:
                        response.error = "Failed to parse JSON from response"

                return response

            except Exception as e:
                last_error = str(e)
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")

                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))

        return LLMResponse(
            content="",
            success=False,
            error=f"All {self.config.max_retries} attempts failed. Last error: {last_error}"
        )

    def _call_anthropic(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Make Anthropic API call."""
        kwargs = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system:
            kwargs["system"] = system

        if temperature > 0:
            kwargs["temperature"] = temperature

        message = self._client.messages.create(**kwargs)

        content = message.content[0].text if message.content else ""

        return LLMResponse(
            content=content,
            model=message.model,
            usage={
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            },
            success=True,
        )

    def _call_local(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Make local LLM API call (Ollama or OpenAI-compatible)."""
        if self.config.provider == LLMProvider.OLLAMA:
            return self._call_ollama(prompt, system, max_tokens, temperature)
        else:
            return self._call_openai_compatible(prompt, system, max_tokens, temperature)

    def _call_ollama(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Make Ollama API call."""
        url = f"{self._base_url}/api/generate"

        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n{prompt}"

        payload = {
            "model": self.config.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            }
        }

        response = self._requests.post(
            url,
            json=payload,
            timeout=self.config.timeout_seconds
        )
        response.raise_for_status()

        data = response.json()
        content = data.get("response", "")

        return LLMResponse(
            content=content,
            model=self.config.model,
            usage={
                "input_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count", 0),
            },
            success=True,
        )

    def _call_openai_compatible(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Make OpenAI-compatible API call (LM Studio, vLLM, etc.)."""
        url = f"{self._base_url}/v1/chat/completions"

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = self._requests.post(
            url,
            json=payload,
            timeout=self.config.timeout_seconds
        )
        response.raise_for_status()

        data = response.json()
        content = ""
        if data.get("choices"):
            content = data["choices"][0].get("message", {}).get("content", "")

        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            model=data.get("model", self.config.model),
            usage={
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
            success=True,
        )

    def _call_openai(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Make OpenAI API call."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        content = response.choices[0].message.content if response.choices else ""

        return LLMResponse(
            content=content,
            model=response.model,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            } if response.usage else None,
            success=True,
        )

    def _parse_json(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON from LLM response, handling common formatting issues.
        """
        if not content:
            return None

        # Clean the response
        cleaned = content.strip()

        # Remove markdown code blocks if present
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first line (```json or ```)
            lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)

        # Try to find JSON object in the text
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from text (find first { to last })
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass

        logger.warning(f"Failed to parse JSON from response: {content[:200]}...")
        return None

    def is_available(self) -> bool:
        """Check if the LLM client is properly configured and available."""
        return self._initialize()

    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the configured model."""
        return {
            "provider": self.config.provider.value,
            "model": self.config.model,
            "available": self.is_available(),
            "max_tokens": self.config.max_tokens,
        }


# Convenience function for quick one-off calls
def quick_llm_call(
    prompt: str,
    system: Optional[str] = None,
    response_format: str = "text",
    config: Optional[Dict[str, Any]] = None,
) -> LLMResponse:
    """
    Quick LLM call without explicit client management.

    Args:
        prompt: The prompt to send
        system: Optional system message
        response_format: "text" or "json"
        config: Optional config dict, uses defaults if not provided

    Returns:
        LLMResponse
    """
    config = config or {}
    client = LLMClient.from_config(config)
    return client.call(prompt, system, response_format)
