"""
providers.py — the LLM backend abstraction for Shifting Truth.

The whole game talks to one interface: LLMProvider.chat().
Swap backends without touching any game logic:

    AnthropicProvider  -> public API (today)
    OnnxProvider       -> local onnxruntime-genai model (later)
    MockProvider       -> no network, for plumbing tests

To add a new backend (llama.cpp, vLLM, OpenAI-compatible server...),
subclass LLMProvider and implement chat().
"""

from __future__ import annotations
import json
import os
import urllib.request
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """messages: [{"role": "user"|"assistant", "content": str}, ...]"""

    @abstractmethod
    def chat(self, system: str, messages: list[dict], max_tokens: int = 300) -> str:
        ...


# ----------------------------------------------------------------
# 1) Public API backend (today)
# ----------------------------------------------------------------
class AnthropicProvider(LLMProvider):
    API_URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.environ.get("ST_MODEL", "claude-haiku-4-5-20251001")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Set ANTHROPIC_API_KEY in your environment, e.g.\n"
                "  export ANTHROPIC_API_KEY=sk-ant-..."
            )

    def chat(self, system: str, messages: list[dict], max_tokens: int = 300) -> str:
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        req = urllib.request.Request(
            self.API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return "".join(
            block.get("text", "") for block in data.get("content", [])
        ).strip()


# ----------------------------------------------------------------
# 2) Local ONNX backend (later) — the swap point you asked for.
#
# Uses onnxruntime-genai, which bundles tokenizer + KV-cache +
# sampling for ONNX LLMs. Setup (one time):
#
#   pip install onnxruntime-genai          # or onnxruntime-genai-cuda
#   # download an ONNX chat model, e.g. Phi-3.5-mini-instruct-onnx
#   # from Hugging Face, then:
#   python main.py --provider onnx --onnx-dir ./phi35-mini-onnx
#
# NOTE: small local models follow persona instructions less
# reliably than frontier API models. Expect to tighten the NPC
# prompts (shorter, more explicit) and lean harder on the referee
# layer when you make this switch.
# ----------------------------------------------------------------
class OnnxProvider(LLMProvider):
    def __init__(self, model_dir: str):
        try:
            import onnxruntime_genai as og  # noqa: lazy import
        except ImportError as e:
            raise RuntimeError(
                "onnxruntime-genai is not installed.\n"
                "  pip install onnxruntime-genai\n"
                "then pass --onnx-dir pointing at an ONNX chat model folder."
            ) from e
        self._og = og
        self.model = og.Model(model_dir)
        self.tokenizer = og.Tokenizer(self.model)

    def _render(self, system: str, messages: list[dict]) -> str:
        # Generic chat template (Phi-style). If your model uses a
        # different template (Llama, Qwen...), adjust here only.
        parts = [f"<|system|>\n{system}<|end|>"]
        for m in messages:
            tag = "user" if m["role"] == "user" else "assistant"
            parts.append(f"<|{tag}|>\n{m['content']}<|end|>")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    def chat(self, system: str, messages: list[dict], max_tokens: int = 300) -> str:
        og = self._og
        prompt = self._render(system, messages)
        params = og.GeneratorParams(self.model)
        params.set_search_options(max_length=4096, temperature=0.8, top_p=0.95)
        generator = og.Generator(self.model, params)
        generator.append_tokens(self.tokenizer.encode(prompt))
        out_tokens = []
        while not generator.is_done() and len(out_tokens) < max_tokens:
            generator.generate_next_token()
            out_tokens.append(generator.get_next_tokens()[0])
        return self.tokenizer.decode(out_tokens).strip()


# ----------------------------------------------------------------
# 3) Mock backend — verifies the whole game loop with no network.
# ----------------------------------------------------------------
class MockProvider(LLMProvider):
    def chat(self, system: str, messages: list[dict], max_tokens: int = 300) -> str:
        name = "The suspect"
        for line in system.splitlines():
            if line.startswith("You are "):
                name = line.removeprefix("You are ").split(",")[0].strip()
                break
        last = messages[-1]["content"] if messages else ""
        return (
            f"({name} studies you for a moment.) \"You ask about "
            f"'{last[:60]}'... I've told you what I know.\" "
            f"[mock reply — run with --provider anthropic for real play]"
        )


def make_provider(kind: str, **kwargs) -> LLMProvider:
    kind = kind.lower()
    if kind == "anthropic":
        return AnthropicProvider(model=kwargs.get("model"))
    if kind == "onnx":
        return OnnxProvider(model_dir=kwargs["onnx_dir"])
    if kind == "mock":
        return MockProvider()
    raise ValueError(f"Unknown provider: {kind}")


def provider_from_config(cfg: dict) -> LLMProvider:
    """Build a provider for one agent from its config block."""
    kind = (cfg or {}).get("provider", "mock").lower()
    if kind == "anthropic":
        return AnthropicProvider(model=cfg.get("model"))
    if kind == "onnx":
        if not cfg.get("onnx_dir"):
            raise ValueError("onnx provider needs 'onnx_dir' in config")
        return OnnxProvider(model_dir=cfg["onnx_dir"])
    if kind == "mock":
        return MockProvider()
    raise ValueError(f"Unknown provider in config: {kind}")
