"""OpenAI Whisper API speech-to-text entity."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterable
import io
import wave

import requests

from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
    SpeechToTextEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    CONF_API_KEY,
    CONF_MODEL,
    CONF_NAME,
    CONF_SOURCE,
    CONF_URL,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import _LOGGER
from .const import (
    CONF_CUSTOM_PROVIDER,
    CONF_OPENROUTER_REQUEST_MODE,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    SUPPORTED_LANGUAGES,
)
from .whisper_provider import WhisperModel, whisper_providers
from .whisper_provider import (
    OPENROUTER_APP_REFERER,
    OPENROUTER_APP_TITLE,
    REQUEST_MODE_AUDIO_TRANSCRIPTIONS,
    REQUEST_MODE_OPENROUTER_AUDIO_TRANSCRIPTIONS,
    REQUEST_MODE_OPENROUTER_CHAT_AUDIO,
)

OPENROUTER_REQUEST_MODES = {
    REQUEST_MODE_OPENROUTER_AUDIO_TRANSCRIPTIONS,
    REQUEST_MODE_OPENROUTER_CHAT_AUDIO,
}

OPENROUTER_TRANSCRIPTION_PROMPT = (
    "Transcribe the provided audio. Return only the spoken transcript text. "
    "Do not add commentary, labels, markdown, or translations."
)

# Mapping of regional language variants to base language codes for Whisper API.
# Whisper only supports base language codes (e.g., "zh"), not regional variants
# (e.g., "zh-tw"). These regional variants are needed for Home Assistant's
# intent recognition system to properly load language-specific intents.
# See: https://github.com/home-assistant/intents/issues/1104
LANGUAGE_TO_WHISPER: dict[str, str] = {
    "zh-cn": "zh",
    "zh-tw": "zh",
    "zh-hk": "zh",
    "zh-hans": "zh",
    "zh-hant": "zh",
}


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Whisper speech platform via config entry."""
    _LOGGER.debug(f"STT setup Entry {config_entry.entry_id}")

    custom = config_entry.data.get(CONF_CUSTOM_PROVIDER, False)
    request_mode = REQUEST_MODE_AUDIO_TRANSCRIPTIONS

    if custom:
        api_url = config_entry.data[CONF_URL]
        model = WhisperModel(config_entry.options[CONF_MODEL], SUPPORTED_LANGUAGES)
    else:
        provider = whisper_providers[config_entry.data[CONF_SOURCE]]
        api_url = provider.url
        request_mode = provider.request_mode
        if provider.request_mode in OPENROUTER_REQUEST_MODES:
            request_mode = config_entry.options.get(
                CONF_OPENROUTER_REQUEST_MODE, REQUEST_MODE_OPENROUTER_CHAT_AUDIO
            )
        if request_mode in (
            REQUEST_MODE_OPENROUTER_AUDIO_TRANSCRIPTIONS,
            REQUEST_MODE_OPENROUTER_CHAT_AUDIO,
        ):
            model = WhisperModel(config_entry.options[CONF_MODEL], SUPPORTED_LANGUAGES)
        else:
            model = provider.models[config_entry.options[CONF_MODEL]]

    async_add_entities([
        OpenAIWhisperCloudEntity(
            custom=custom,
            request_mode=request_mode,
            api_url=api_url,
            api_key=config_entry.data.get(CONF_API_KEY, ""),
            model=model,
            temperature=config_entry.options[CONF_TEMPERATURE],
            prompt=config_entry.options[CONF_PROMPT],
            name=config_entry.data[CONF_NAME],
            unique_id=config_entry.entry_id
        )
    ])



class OpenAIWhisperCloudEntity(SpeechToTextEntity):
    """OpenAI Whisper API provider entity."""

    def __init__(
        self,
        custom: bool,
        request_mode: str,
        api_url: str,
        api_key: str,
        model: WhisperModel,
        temperature,
        prompt,
        name,
        unique_id,
    ) -> None:
        """Init STT service."""
        self.custom = custom
        self.request_mode = request_mode
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.prompt = prompt
        self._attr_name = name
        self._attr_unique_id = unique_id

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return self.model.languages

    @property
    def supported_formats(self) -> list[AudioFormats]:
        """Return a list of supported formats."""
        return [AudioFormats.WAV]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        """Return a list of supported codecs."""
        return [AudioCodecs.PCM]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        """Return a list of supported bit rates."""
        return [
            AudioBitRates.BITRATE_8,
            AudioBitRates.BITRATE_16,
            AudioBitRates.BITRATE_24,
            AudioBitRates.BITRATE_32,
        ]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        """Return a list of supported sample rates."""
        return [
            AudioSampleRates.SAMPLERATE_8000,
            AudioSampleRates.SAMPLERATE_16000,
            AudioSampleRates.SAMPLERATE_44100,
            AudioSampleRates.SAMPLERATE_48000,
        ]

    @property
    def supported_channels(self) -> list[AudioChannels]:
        """Return a list of supported channels."""
        return [AudioChannels.CHANNEL_MONO, AudioChannels.CHANNEL_STEREO]

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> SpeechResult:
        """Process an audio stream to STT service."""

        _LOGGER.debug("Processing audio stream: %s", metadata)

        data = b""
        async for chunk in stream:
            data += chunk
            if len(data) / (1024 * 1024) > 24.5:
                _LOGGER.error("Audio stream size exceed the maximum allowed by OpenAI which is 25Mb")
                return SpeechResult("", SpeechResultState.ERROR)

        if not data:
            _LOGGER.error("No audio data received")
            return SpeechResult("", SpeechResultState.ERROR)

        try:
            temp_file = io.BytesIO()
            with wave.open(temp_file, "wb") as wav_file:
                wav_file.setnchannels(metadata.channel)
                wav_file.setframerate(metadata.sample_rate)
                wav_file.setsampwidth(2)
                wav_file.writeframes(data)

            # Ensure the buffer is at the start before passing it
            temp_file.seek(0)

            _LOGGER.debug("Temp wav audio file created of %.2f Mb", temp_file.getbuffer().nbytes / (1024 * 1024))

            if self.request_mode == REQUEST_MODE_OPENROUTER_CHAT_AUDIO:
                return await self._async_process_openrouter_audio(metadata, temp_file)

            if self.request_mode == REQUEST_MODE_OPENROUTER_AUDIO_TRANSCRIPTIONS:
                return await self._async_process_openrouter_transcription_audio(
                    metadata, temp_file
                )

            return await self._async_process_transcriptions_audio(metadata, temp_file)

        except requests.exceptions.RequestException as e:
            _LOGGER.error(e)
            return SpeechResult("", SpeechResultState.ERROR)

    async def _async_process_transcriptions_audio(
        self, metadata: SpeechMetadata, temp_file: io.BytesIO
    ) -> SpeechResult:
        """Process audio through an OpenAI-compatible transcriptions endpoint."""
        files = {
            "file": ("audio.wav", temp_file, "audio/wav"),
        }

        # Convert regional language variants to base language for Whisper API.
        whisper_language = LANGUAGE_TO_WHISPER.get(
            metadata.language.lower() if metadata.language else "",
            metadata.language,
        )
        if whisper_language != metadata.language:
            _LOGGER.debug(
                "Converted language '%s' to '%s' for Whisper API",
                metadata.language,
                whisper_language,
            )

        data = {
            "model": self.model.name,
            "language": whisper_language,
            "temperature": self.temperature,
            "prompt": self.prompt,
            "response_format": "json",
        }

        response = await asyncio.to_thread(
            requests.post,
            f"{self.api_url}/v1/audio/transcriptions" if not self.custom else self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
            },
            files=files,
            data=data,
        )

        _LOGGER.debug(
            "Transcription request took %f s and returned %d - %s",
            response.elapsed.total_seconds(),
            response.status_code,
            response.reason,
        )

        if response.status_code < 200 or response.status_code >= 300:
            _LOGGER.error(response.text)
            return SpeechResult("", SpeechResultState.ERROR)

        try:
            transcription = response.json().get("text", "")
        except (AttributeError, ValueError) as e:
            _LOGGER.error("Transcription response was not valid JSON: %s", e)
            return SpeechResult("", SpeechResultState.ERROR)

        _LOGGER.debug("TRANSCRIPTION: %s", transcription)

        if not transcription:
            _LOGGER.error(response.text)
            return SpeechResult("", SpeechResultState.ERROR)

        return SpeechResult(transcription, SpeechResultState.SUCCESS)

    async def _async_process_openrouter_transcription_audio(
        self, metadata: SpeechMetadata, temp_file: io.BytesIO
    ) -> SpeechResult:
        """Process audio through OpenRouter's dedicated transcription endpoint."""
        openrouter_language = LANGUAGE_TO_WHISPER.get(
            metadata.language.lower() if metadata.language else "",
            metadata.language,
        )
        if openrouter_language != metadata.language:
            _LOGGER.debug(
                "Converted language '%s' to '%s' for OpenRouter STT API",
                metadata.language,
                openrouter_language,
            )

        payload = {
            "model": self.model.name,
            "input_audio": {
                "data": base64.b64encode(temp_file.getvalue()).decode("ascii"),
                "format": "wav",
            },
            "temperature": self.temperature,
        }
        if openrouter_language:
            payload["language"] = openrouter_language

        response = await asyncio.to_thread(
            requests.post,
            f"{self.api_url}/v1/audio/transcriptions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": OPENROUTER_APP_REFERER,
                "X-OpenRouter-Title": OPENROUTER_APP_TITLE,
            },
            json=payload,
        )

        _LOGGER.debug(
            "OpenRouter STT request took %f s and returned %d - %s",
            response.elapsed.total_seconds(),
            response.status_code,
            response.reason,
        )

        if response.status_code < 200 or response.status_code >= 300:
            _LOGGER.error(response.text)
            return SpeechResult("", SpeechResultState.ERROR)

        try:
            transcription = response.json().get("text", "")
        except (AttributeError, ValueError) as e:
            _LOGGER.error("OpenRouter STT response was not valid JSON: %s", e)
            return SpeechResult("", SpeechResultState.ERROR)

        if not isinstance(transcription, str):
            _LOGGER.error("OpenRouter STT response text was not a string")
            return SpeechResult("", SpeechResultState.ERROR)

        transcription = transcription.strip()
        _LOGGER.debug("TRANSCRIPTION: %s", transcription)

        if not transcription:
            _LOGGER.error(response.text)
            return SpeechResult("", SpeechResultState.ERROR)

        return SpeechResult(transcription, SpeechResultState.SUCCESS)

    async def _async_process_openrouter_audio(
        self, metadata: SpeechMetadata, temp_file: io.BytesIO
    ) -> SpeechResult:
        """Process audio through OpenRouter chat completions."""
        prompt = OPENROUTER_TRANSCRIPTION_PROMPT
        if metadata.language:
            prompt = f"{prompt}\nExpected language: {metadata.language}."
        if self.prompt:
            prompt = f"{prompt}\nContext/glossary: {self.prompt}"

        payload = {
            "model": self.model.name,
            "temperature": self.temperature,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(
                                    temp_file.getvalue()
                                ).decode("ascii"),
                                "format": "wav",
                            },
                        },
                    ],
                }
            ],
        }

        response = await asyncio.to_thread(
            requests.post,
            f"{self.api_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": OPENROUTER_APP_REFERER,
                "X-OpenRouter-Title": OPENROUTER_APP_TITLE,
            },
            json=payload,
        )

        _LOGGER.debug(
            "OpenRouter transcription request took %f s and returned %d - %s",
            response.elapsed.total_seconds(),
            response.status_code,
            response.reason,
        )

        if response.status_code < 200 or response.status_code >= 300:
            _LOGGER.error(response.text)
            return SpeechResult("", SpeechResultState.ERROR)

        try:
            response_body = response.json()
            transcription = response_body["choices"][0]["message"].get("content", "")
        except (AttributeError, ValueError, KeyError, IndexError, TypeError) as e:
            _LOGGER.error("OpenRouter transcription response was invalid: %s", e)
            return SpeechResult("", SpeechResultState.ERROR)

        if not isinstance(transcription, str):
            _LOGGER.error("OpenRouter transcription response content was not text")
            return SpeechResult("", SpeechResultState.ERROR)

        transcription = transcription.strip()
        _LOGGER.debug("TRANSCRIPTION: %s", transcription)

        if not transcription:
            _LOGGER.error(response.text)
            return SpeechResult("", SpeechResultState.ERROR)

        return SpeechResult(transcription, SpeechResultState.SUCCESS)
