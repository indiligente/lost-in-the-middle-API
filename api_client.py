"""Cliente comum para Groq e OpenRouter. Compativel com Python 3.9."""
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class Resultado:
    provider: str
    requested_model: str
    returned_model: Optional[str]
    response_id: Optional[str]
    text: str
    finish_reason: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    duration_seconds: float
    backend_provider: Optional[str]

    def to_dict(self):
        return asdict(self)


class ClienteAPI:
    def __init__(self, provider, model, temperature=0, top_p=1.0,
                 max_output_tokens=200, timeout_seconds=120):
        if provider not in ("groq", "openrouter"):
            raise ValueError("provider deve ser groq ou openrouter.")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Informe o ID do modelo em config.json ou --model.")
        if max_output_tokens < 1 or timeout_seconds <= 0:
            raise ValueError("Limite de tokens e timeout devem ser positivos.")

        variable = "GROQ_API_KEY" if provider == "groq" else "OPENROUTER_API_KEY"
        key = os.environ.get(variable)
        if not key:
            raise ValueError("Configure {} no .env ou no ambiente.".format(variable))

        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        # Zero retries: cada execucao inicial realiza uma unica tentativa.
        if provider == "groq":
            from groq import Groq
            self.client = Groq(api_key=key, timeout=timeout_seconds, max_retries=0)
        else:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=key, base_url="https://openrouter.ai/api/v1",
                timeout=timeout_seconds, max_retries=0,
            )

    def gerar(self, prompt):
        started = time.perf_counter()
        request = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_output_tokens,
        }
        if self.provider == "groq" and self.model in (
            "openai/gpt-oss-20b", "openai/gpt-oss-120b",
        ):
            # O avaliador espera somente a resposta final, sem o raciocinio.
            request["include_reasoning"] = False

        response = self.client.chat.completions.create(
            **request
        )
        elapsed = time.perf_counter() - started
        if not response.choices:
            raise RuntimeError("API retornou uma resposta sem choices.")
        choice = response.choices[0]
        usage = response.usage
        message = choice.message
        text = getattr(message, "content", None)
        if not text:
            # Alguns modelos de raciocinio podem consumir o limite antes de
            # preencher content; preservar reasoning evita uma resposta vazia.
            text = getattr(message, "reasoning", None) or ""
        return Resultado(
            provider=self.provider, requested_model=self.model,
            returned_model=getattr(response, "model", None),
            response_id=getattr(response, "id", None),
            text=text,
            finish_reason=choice.finish_reason,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            duration_seconds=elapsed,
            backend_provider=getattr(response, "provider", None),
        )

    def close(self):
        self.client.close()
