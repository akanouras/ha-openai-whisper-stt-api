"""OpenAI Whisper API Providers."""

from .const import SUPPORTED_LANGUAGES

REQUEST_MODE_AUDIO_TRANSCRIPTIONS = "audio_transcriptions"
REQUEST_MODE_OPENROUTER_AUDIO_TRANSCRIPTIONS = "openrouter_audio_transcriptions"
REQUEST_MODE_OPENROUTER_CHAT_AUDIO = "openrouter_chat_audio"

OPENROUTER_APP_REFERER = (
    "https://github.com/fabio-garavini/ha-openai-whisper-stt-api"
)
OPENROUTER_APP_TITLE = "OpenAI Whisper Cloud for Home Assistant"

OPENROUTER_LEGACY_MODEL_NOTE = (
    "* Models use OpenRouter chat completions audio because they do not "
    "support the transcription endpoint."
)

OPENROUTER_STT_FALLBACK_MODEL_NAMES = [
    "openai/gpt-4o-mini-transcribe",
    "openai/gpt-4o-transcribe",
    "openai/whisper-1",
    "openai/whisper-large-v3",
    "openai/whisper-large-v3-turbo",
    "mistralai/voxtral-mini-transcribe",
    "nvidia/parakeet-tdt-0.6b-v3",
    "google/chirp-3",
    "qwen/qwen3-asr-flash-2026-02-10",
]

OPENROUTER_LEGACY_FALLBACK_MODEL_NAMES = [
    "openai/gpt-audio-mini",
    "openai/gpt-audio",
    "openai/gpt-4o-audio-preview",
    "mistralai/voxtral-small-24b-2507",
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.0-flash-lite-001",
]


class WhisperModel:
    """Whisper Model."""

    def __init__(
        self,
        name: str,
        languages: list,
        label: str | None = None,
        request_mode: str | None = None,
    ) -> None:
        """Init."""
        self.name = name
        self.languages = languages
        self.label = label or name
        self.request_mode = request_mode


class WhisperProvider:
    """Whisper API Provider."""

    def __init__(
        self,
        name: str,
        url: str,
        models: list,
        default_model: int,
        request_mode: str = REQUEST_MODE_AUDIO_TRANSCRIPTIONS,
    ) -> None:
        """Init."""
        self.name = name
        self.url = url
        self.models = models
        self.default_model = default_model
        self.request_mode = request_mode


whisper_providers = [
    WhisperProvider(
        "OpenAI",
        "https://api.openai.com",
        [
            WhisperModel("whisper-1", SUPPORTED_LANGUAGES),
            WhisperModel("gpt-4o-transcribe", SUPPORTED_LANGUAGES),
            WhisperModel("gpt-4o-mini-transcribe", SUPPORTED_LANGUAGES),
        ],
        2
    ),
    WhisperProvider(
        "GroqCloud",
        "https://api.groq.com/openai",
        [
            WhisperModel("whisper-large-v3", SUPPORTED_LANGUAGES),
            WhisperModel("whisper-large-v3-turbo", SUPPORTED_LANGUAGES)
        ],
        1
    ),
    WhisperProvider(
        "Mistral AI",
        "https://api.mistral.ai",
        [
            WhisperModel("voxtral-mini-latest", languages = ["en", "fr", "de", "es", "it", "pt", "nl", "hi", "ar"])
        ],
        0
    ),
    WhisperProvider(
        "OpenRouter",
        "https://openrouter.ai/api",
        [
            *[
                WhisperModel(
                    model_name,
                    SUPPORTED_LANGUAGES,
                    request_mode=REQUEST_MODE_OPENROUTER_AUDIO_TRANSCRIPTIONS,
                )
                for model_name in OPENROUTER_STT_FALLBACK_MODEL_NAMES
            ],
            *[
                WhisperModel(
                    model_name,
                    SUPPORTED_LANGUAGES,
                    f"{model_name} *",
                    REQUEST_MODE_OPENROUTER_CHAT_AUDIO,
                )
                for model_name in OPENROUTER_LEGACY_FALLBACK_MODEL_NAMES
            ],
        ],
        0,
        REQUEST_MODE_OPENROUTER_AUDIO_TRANSCRIPTIONS,
    ),
    WhisperProvider("Custom", "", [], 0),
]
