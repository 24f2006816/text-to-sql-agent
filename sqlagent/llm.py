"""LLM back ends. Anything with `generate(messages) -> str` works, so tests use a scripted fake."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Protocol

Message = dict[str, str]

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


class LLM(Protocol):
    name: str

    def generate(self, messages: list[Message]) -> str: ...


class TransformersLLM:
    """Runs an open model locally with Hugging Face transformers (CPU or GPU), greedy decoding."""

    def __init__(self, model_id: str = DEFAULT_MODEL, max_new_tokens: int = 256, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_id
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(self.device)
        self.model.eval()
        self._torch = torch

    def generate(self, messages: list[Message]) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


class OpenAICompatibleLLM:
    """Any OpenAI-compatible chat endpoint (OpenAI, Groq, Together, a local vLLM or Ollama server)."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        self.name = model
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")

    def generate(self, messages: list[Message]) -> str:
        body = json.dumps({"model": self.name, "messages": messages, "temperature": 0}).encode()
        if not self.base_url.startswith(("https://", "http://")):
            raise ValueError("OPENAI_BASE_URL must be an http(s) URL")
        request = urllib.request.Request(  # noqa: S310 - scheme checked above
            f"{self.base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - configured endpoint
            return json.load(response)["choices"][0]["message"]["content"]


def build_llm(spec: str | None = None) -> LLM:
    """`hf:<model id>` (default) or `openai:<model>`; also read from the SQLAGENT_MODEL variable."""
    spec = spec or os.getenv("SQLAGENT_MODEL") or f"hf:{DEFAULT_MODEL}"
    kind, _, model = spec.partition(":")
    if kind == "openai":
        return OpenAICompatibleLLM(model)
    if kind == "hf":
        return TransformersLLM(model or DEFAULT_MODEL)
    return TransformersLLM(spec)
