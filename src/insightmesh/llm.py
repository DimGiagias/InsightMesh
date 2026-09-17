from functools import lru_cache
import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_PROVIDER = "openai"

DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "gemini": "gemini-3.1-flash-lite",
    "openrouter": "google/gemma-4-31b-it:free",
    "ollama": "llama3.1",
}


def _default_temperature() -> float:
    return float(os.getenv("LLM_TEMPERATURE", "0"))


def _default_model(provider: str, from_env_provider: bool) -> str:
    model = os.getenv(f"{provider.upper()}_MODEL")
    
    if not model and from_env_provider:
        model = os.getenv("LLM_MODEL")
    return model or DEFAULT_MODELS[provider]


def _build_openai(model: str, temperature: float, **kwargs):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, temperature=temperature, **kwargs)


def _build_gemini(model: str, temperature: float, **kwargs):
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        kwargs.setdefault("google_api_key", api_key)

    return ChatGoogleGenerativeAI(model=model, temperature=temperature, **kwargs)


def _build_openrouter(model: str, temperature: float, **kwargs):
    from langchain_openai import ChatOpenAI

    kwargs.setdefault(
        "base_url", os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    )
    kwargs.setdefault("api_key", os.getenv("OPENROUTER_API_KEY"))

    return ChatOpenAI(model=model, temperature=temperature, **kwargs)


def _build_ollama(model: str, temperature: float, **kwargs):
    from langchain_ollama import ChatOllama

    kwargs.setdefault(
        "base_url", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )

    return ChatOllama(model=model, temperature=temperature, **kwargs)


BUILDERS = {
    "openai": _build_openai,
    "gemini": _build_gemini,
    "openrouter": _build_openrouter,
    "ollama": _build_ollama,
}


def _normalize(provider: str | None) -> str:
    provider = (provider or os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).lower().strip()
    provider = {
        "google": "gemini",
        "open_router": "openrouter",
        "local": "ollama",
    }.get(provider, provider)

    if provider not in BUILDERS:
        raise ValueError(
            f"Unknown LLM provider {provider!r}. Choose one of: {', '.join(BUILDERS)}"
        )
    return provider


@lru_cache(maxsize=None)
def _cached_llm(provider: str, model: str, temperature: float, kwargs: tuple):
    return BUILDERS[provider](model, temperature, **dict(kwargs))


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    **kwargs,
):
    """
    Returns a chat model based on the selected provider.

    Models are cached, so repeated calls with the same arguments reuse one client.
    Calls with unhashable kwargs (e.g. a dict of headers) build a fresh client.
    """
    from_env_provider = provider is None
    provider = _normalize(provider)
    model = model or _default_model(provider, from_env_provider)
    temperature = _default_temperature() if temperature is None else temperature

    try:
        return _cached_llm(provider, model, temperature, tuple(sorted(kwargs.items())))
    except TypeError as e:
        if "unhashable" not in str(e):
            raise
        return BUILDERS[provider](model, temperature, **kwargs)


def get_structured_llm(schema, llm=None, method: str | None = None):
    """
    Bind a schema to a model, using the structured-output method that fits.

    Providers disagree on how structured output is produced: OpenAI and
    OpenRouter default to tool calling, while many local Ollama models only
    handle a JSON schema. Override with ``LLM_STRUCTURED_OUTPUT_METHOD``
    (``function_calling``, ``json_schema`` or ``json_mode``).
    """
    llm = llm if llm is not None else get_llm()
    method = method or os.getenv("LLM_STRUCTURED_OUTPUT_METHOD")

    if method:
        return llm.with_structured_output(schema, method=method)
    return llm.with_structured_output(schema)


def __getattr__(name: str):
    if name == "llm":
        return get_llm()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
