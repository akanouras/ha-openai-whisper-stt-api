"""Config flow for OpenAI Whisper Cloud integration."""

from __future__ import annotations

import asyncio
from typing import Any

import requests
import voluptuous as vol

from homeassistant import exceptions
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlowWithConfigEntry,
)
from homeassistant.const import (
    CONF_API_KEY,
    CONF_MODEL,
    CONF_NAME,
    CONF_SOURCE,
    CONF_URL,
)
from homeassistant.core import callback
import homeassistant.helpers.config_validation as cv
from homeassistant.helpers.selector import (
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
)

from .const import (
    _LOGGER,
    CONF_CUSTOM_PROVIDER,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DOMAIN,
    SUPPORTED_LANGUAGES,
)
from .whisper_provider import (
    REQUEST_MODE_OPENROUTER_CHAT_AUDIO,
    WhisperModel,
    WhisperProvider,
    whisper_providers,
)

REQUEST_TIMEOUT = 30

PROVIDER_SELECTION_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_SOURCE, default="0"): SelectSelector(
            SelectSelectorConfig(
                options=[
                    SelectOptionDict(value=str(i), label=p.name)
                    for i, p in enumerate(whisper_providers)
                ],
                mode=SelectSelectorMode.DROPDOWN,
            )
        ),
    }
)


def _is_openrouter_provider(provider: WhisperProvider) -> bool:
    """Return true if the provider uses OpenRouter chat audio requests."""
    return provider.request_mode == REQUEST_MODE_OPENROUTER_CHAT_AUDIO


def _stored_model_value(provider: WhisperProvider, model_name: str) -> int | str:
    """Return the config entry model value for a provider."""
    if _is_openrouter_provider(provider):
        return model_name

    return [x.name for x in provider.models].index(model_name)


def _entry_model_name(entry: ConfigEntry, provider: WhisperProvider) -> str:
    """Return a config entry's selected model name."""
    model = entry.options.get(CONF_MODEL)
    if _is_openrouter_provider(provider):
        return model or provider.models[provider.default_model].name

    if model is None:
        return provider.models[provider.default_model].name

    return provider.models[model].name


def _include_selected_model(
    models: list[WhisperModel], selected_model: str | None
) -> list[WhisperModel]:
    """Keep a previously saved OpenRouter model selectable."""
    if not selected_model or selected_model in [model.name for model in models]:
        return models

    return [
        WhisperModel(selected_model, SUPPORTED_LANGUAGES, f"{selected_model} (saved)"),
        *models,
    ]


async def async_get_openrouter_models(
    provider: WhisperProvider,
) -> tuple[list[WhisperModel], bool]:
    """Fetch OpenRouter models that accept audio and return text."""
    try:
        response = await asyncio.to_thread(
            requests.get,
            url=f"{provider.url}/v1/models",
            timeout=REQUEST_TIMEOUT,
        )

        _LOGGER.debug(
            "OpenRouter models request took %f s and returned %d - %s",
            response.elapsed.total_seconds(),
            response.status_code,
            response.reason,
        )

        if response.status_code != 200:
            _LOGGER.warning(
                "OpenRouter models request failed with status %d; using fallback models",
                response.status_code,
            )
            return provider.models, False

        model_data = response.json().get("data", [])
        models: list[WhisperModel] = []
        seen: set[str] = set()

        for model in model_data:
            model_id = model.get("id")
            architecture = model.get("architecture") or {}
            input_modalities = architecture.get("input_modalities") or []
            output_modalities = architecture.get("output_modalities") or []

            if (
                not model_id
                or model_id in seen
                or "audio" not in input_modalities
                or "text" not in output_modalities
            ):
                continue

            seen.add(model_id)
            model_name = model.get("name")
            label = (
                f"{model_name} ({model_id})"
                if model_name and model_name != model_id
                else model_id
            )
            models.append(WhisperModel(model_id, SUPPORTED_LANGUAGES, label))

        if not models:
            _LOGGER.warning(
                "OpenRouter models response contained no audio-to-text models; using fallback models"
            )
            return provider.models, False

        return models, True

    except (
        AttributeError,
        requests.exceptions.RequestException,
        ValueError,
        TypeError,
    ) as e:
        _LOGGER.warning("Unable to fetch OpenRouter models; using fallback models: %s", e)
        return provider.models, False


async def async_model_schema_entry(
    provider: WhisperProvider, selected_model: str | None = None
) -> tuple[vol.Required, vol.In | SelectSelector]:
    """Build the model schema entry for a provider."""
    if _is_openrouter_provider(provider):
        models, _models_available = await async_get_openrouter_models(provider)
        models = _include_selected_model(models, selected_model)
        default_model = selected_model or provider.models[provider.default_model].name

        return vol.Required(CONF_MODEL, default=default_model), SelectSelector(
            SelectSelectorConfig(
                options=[
                    SelectOptionDict(value=model.name, label=model.label)
                    for model in models
                ],
                mode=SelectSelectorMode.DROPDOWN,
            )
        )

    return (
        vol.Required(
            CONF_MODEL,
            default=selected_model or provider.models[provider.default_model].name,
        ),
        vol.In([x.name for x in provider.models]),
    )


async def validate_input(data: dict, provider: WhisperProvider):
    """Validate the user input."""

    obscured_api_key = data.get(CONF_API_KEY)
    data[CONF_API_KEY] = "<api_key>"
    _LOGGER.debug("User validation got: %s", data)
    data[CONF_API_KEY] = obscured_api_key

    if data.get(CONF_TEMPERATURE) is None:
        data[CONF_TEMPERATURE] = DEFAULT_TEMPERATURE
    if data.get(CONF_PROMPT) is None:
        data[CONF_PROMPT] = DEFAULT_PROMPT

    if _is_openrouter_provider(provider):
        if not data.get(CONF_API_KEY):
            raise InvalidAPIKey

        response = await asyncio.to_thread(
            requests.get,
            url=f"{provider.url}/v1/key",
            headers={"Authorization": f"Bearer {data.get(CONF_API_KEY)}"},
            timeout=REQUEST_TIMEOUT,
        )

        _LOGGER.debug(
            "OpenRouter key request took %f s and returned %d - %s",
            response.elapsed.total_seconds(),
            response.status_code,
            response.reason,
        )

        if response.status_code == 401:
            raise InvalidAPIKey

        if response.status_code == 403:
            raise UnauthorizedError

        if response.status_code != 200:
            raise UnknownError

        models, models_available = await async_get_openrouter_models(provider)
        if models_available and data.get(CONF_MODEL) not in [
            model.name for model in models
        ]:
            raise WhisperModelNotFound

        _LOGGER.debug("OpenRouter user validation successful")
        return

    response = await asyncio.to_thread(
        requests.get,
        url=f"{provider.url}/v1/models/{data.get(CONF_MODEL)}",
        headers={"Authorization": f"Bearer {data.get(CONF_API_KEY)}"},
        timeout=REQUEST_TIMEOUT,
    )

    _LOGGER.debug(
        "Models request took %f s and returned %d - %s",
        response.elapsed.seconds,
        response.status_code,
        response.reason,
    )

    if response.status_code == 401:
        raise InvalidAPIKey

    if response.status_code == 403:
        raise UnauthorizedError

    if response.status_code == 404:
        raise WhisperModelNotFound

    if response.status_code != 200:
        raise UnknownError

    _LOGGER.debug("User validation successful")


class OptionsFlowHandler(OptionsFlowWithConfigEntry):
    """Handle OpenAI Whisper Cloud options."""

    config_entry: ConfigEntry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the OpenAI options."""

        errors = {}
        custom_provider = self.config_entry.data.get(CONF_CUSTOM_PROVIDER)
        provider = None
        if not custom_provider:
            provider = whisper_providers[self.config_entry.data[CONF_SOURCE]]

        if user_input is not None:
            return self.async_create_entry(
                title=self.config_entry.title,
                data={
                    CONF_MODEL: user_input[CONF_MODEL]
                    if custom_provider
                    else _stored_model_value(provider, user_input[CONF_MODEL]),
                    CONF_TEMPERATURE: user_input[CONF_TEMPERATURE],
                    CONF_PROMPT: user_input.get(CONF_PROMPT, DEFAULT_PROMPT),
                },
            )

        if custom_provider:
            model_schema_key = vol.Required(CONF_MODEL)
            model_schema_value = cv.string
            suggested_model = self.config_entry.options[CONF_MODEL]
        else:
            suggested_model = _entry_model_name(self.config_entry, provider)
            model_schema_key, model_schema_value = await async_model_schema_entry(
                provider, suggested_model
            )

        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                data_schema=vol.Schema(
                    {
                        model_schema_key: model_schema_value,
                        vol.Optional(CONF_TEMPERATURE): vol.All(
                            vol.Coerce(float), vol.Range(min=0, max=1)
                        ),
                        vol.Optional(CONF_PROMPT): cv.string,
                    }
                ),
                suggested_values={
                    CONF_MODEL: suggested_model,
                    CONF_TEMPERATURE: self.config_entry.options[CONF_TEMPERATURE],
                    CONF_PROMPT: self.config_entry.options.get(CONF_PROMPT, DEFAULT_PROMPT),
                },
            ),
            errors=errors,
        )


class ConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle UI config flow."""

    VERSION = 1
    MINOR_VERSION = 3

    _provider: WhisperProvider | None = None

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlowHandler:
        """Options callback for Reolink."""
        return OptionsFlowHandler(config_entry)

    async def async_step_user(
        self,
        user_input: dict[str, Any] | None = None,
        errors: dict[str, str] | None = None,
    ) -> ConfigFlowResult:
        """Handle initial step."""
        errors = {}
        if user_input is not None:
            if int(user_input[CONF_SOURCE]) != len(whisper_providers) -1:
                self._provider = whisper_providers[int(user_input[CONF_SOURCE])]

            return await self.async_step_whisper()

        return self.async_show_form(
            step_id="user", data_schema=PROVIDER_SELECTION_SCHEMA, errors=errors
        )

    async def async_step_whisper(
        self,
        user_input: dict[str, Any] | None = None,
        errors: dict[str, str] | None = None,
    ) -> ConfigFlowResult:
        """Handle initial step."""
        errors = {}
        if user_input is not None:

            if self._provider is None:
                return self.async_create_entry(
                    title=user_input.get(CONF_NAME),
                    data={
                        CONF_CUSTOM_PROVIDER: True,
                        CONF_NAME: user_input[CONF_NAME],
                        CONF_URL: user_input[CONF_URL],
                        CONF_API_KEY: user_input.get(CONF_API_KEY),
                    },
                    options={
                        CONF_MODEL: user_input[CONF_MODEL],
                        CONF_TEMPERATURE: user_input[CONF_TEMPERATURE],
                        CONF_PROMPT: user_input.get(CONF_PROMPT, ""),
                    },
                )

            try:
                await validate_input(user_input, self._provider)

                return self.async_create_entry(
                    title=user_input.get(CONF_NAME),
                    data={
                        CONF_SOURCE: whisper_providers.index(self._provider),
                        CONF_NAME: user_input[CONF_NAME],
                        CONF_API_KEY: user_input[CONF_API_KEY],
                    },
                    options={
                        CONF_MODEL: _stored_model_value(
                            self._provider, user_input[CONF_MODEL]
                        ),
                        CONF_TEMPERATURE: user_input[CONF_TEMPERATURE],
                        CONF_PROMPT: user_input.get(CONF_PROMPT, DEFAULT_PROMPT),
                    },
                )

            except requests.exceptions.RequestException as e:
                _LOGGER.error(e)
                errors["base"] = "connection_error"
            except UnauthorizedError:
                errors["base"] = "unauthorized"
            except InvalidAPIKey:
                errors[CONF_API_KEY] = "invalid_api_key"
            except WhisperModelNotFound:
                errors["base"] = "whisper_not_found"
            except UnknownError:
                errors["base"] = "unknown"

        if self._provider is None:
            return self.async_show_form(
                step_id="whisper",
                data_schema=vol.Schema(
                    {
                        vol.Required(
                            CONF_NAME, default="Custom Whisper"
                        ): cv.string,
                        vol.Required(CONF_URL): cv.string,
                        vol.Optional(CONF_API_KEY): cv.string,
                        vol.Required(CONF_MODEL): cv.string,
                        vol.Optional(
                            CONF_TEMPERATURE, default=DEFAULT_TEMPERATURE
                        ): vol.All(vol.Coerce(float), vol.Range(min=0, max=1)),
                        vol.Optional(CONF_PROMPT): cv.string,
                    }
                ),
                errors=errors,
            )

        model_schema_key, model_schema_value = await async_model_schema_entry(
            self._provider
        )

        return self.async_show_form(
            step_id="whisper",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_NAME, default=f"{self._provider.name} Whisper"
                    ): cv.string,
                    vol.Required(CONF_API_KEY): cv.string,
                    model_schema_key: model_schema_value,
                    vol.Optional(
                        CONF_TEMPERATURE, default=DEFAULT_TEMPERATURE
                    ): vol.All(vol.Coerce(float), vol.Range(min=0, max=1)),
                    vol.Optional(CONF_PROMPT): cv.string,
                }
            ),
            errors=errors,
        )

    async def async_step_reconfigure(
        self,
        user_input: dict[str, Any] | None = None,
        errors: dict[str, str] | None = None,
    ):
        """Config Flow reconfiguration."""
        errors = {}

        entry = self.hass.config_entries.async_get_entry(self.context["entry_id"])

        if entry.data.get(CONF_CUSTOM_PROVIDER, False):

            if user_input is not None:

                self.hass.config_entries.async_update_entry(
                    entry=entry,
                    title=user_input.get(CONF_NAME),
                    data={
                        CONF_CUSTOM_PROVIDER: True,
                        CONF_NAME: user_input[CONF_NAME],
                        CONF_URL: user_input[CONF_URL],
                        CONF_API_KEY: user_input.get(CONF_API_KEY, entry.data.get(CONF_API_KEY, "")),
                    },
                    options={
                        CONF_MODEL: user_input[CONF_MODEL],
                        CONF_TEMPERATURE: user_input[CONF_TEMPERATURE],
                        CONF_PROMPT: user_input.get(CONF_PROMPT, DEFAULT_PROMPT),
                    },
                )

                await self.hass.config_entries.async_reload(self.context["entry_id"])
                return self.async_abort(reason="reconfigure_successful")


            return self.async_show_form(
                step_id="reconfigure",
                data_schema=self.add_suggested_values_to_schema(
                    data_schema=vol.Schema(
                        {
                            vol.Required(
                                CONF_NAME, default="Custom Whisper"
                            ): cv.string,
                            vol.Required(CONF_URL): cv.string,
                            vol.Optional(CONF_API_KEY): cv.string,
                            vol.Required(CONF_MODEL): cv.string,
                            vol.Optional(
                                CONF_TEMPERATURE, default=DEFAULT_TEMPERATURE
                            ): vol.All(vol.Coerce(float), vol.Range(min=0, max=1)),
                            vol.Optional(CONF_PROMPT): cv.string,
                        }
                    ),
                    suggested_values={
                        CONF_NAME: entry.data.get(CONF_NAME),
                        CONF_URL: entry.data.get(CONF_URL),
                        CONF_MODEL: entry.options.get(CONF_MODEL),
                        CONF_TEMPERATURE: entry.options.get(CONF_TEMPERATURE),
                        CONF_PROMPT: entry.options.get(CONF_PROMPT),
                    },
                ),
                errors=errors,
            )

        provider: WhisperProvider = whisper_providers[entry.data.get(CONF_SOURCE)]
        selected_model = _entry_model_name(entry, provider)

        if user_input is not None:

            try:
                await validate_input(
                    {
                        **user_input,
                        CONF_API_KEY: user_input.get(CONF_API_KEY, entry.data.get(CONF_API_KEY, ""))
                    },
                    provider
                )

                self.hass.config_entries.async_update_entry(
                    entry=entry,
                    title=user_input[CONF_NAME],
                    data={
                        CONF_SOURCE: entry.data.get(CONF_SOURCE),
                        CONF_NAME: user_input[CONF_NAME],
                        CONF_API_KEY: user_input.get(CONF_API_KEY, entry.data.get(CONF_API_KEY, "")),
                    },
                    options={
                        CONF_MODEL: _stored_model_value(
                            provider, user_input[CONF_MODEL]
                        ),
                        CONF_TEMPERATURE: user_input[CONF_TEMPERATURE],
                        CONF_PROMPT: user_input.get(CONF_PROMPT, DEFAULT_PROMPT),
                    },
                )
                await self.hass.config_entries.async_reload(self.context["entry_id"])
                return self.async_abort(reason="reconfigure_successful")

            except requests.exceptions.RequestException as e:
                _LOGGER.error(e)
                errors["base"] = "connection_error"
            except UnauthorizedError:
                errors["base"] = "unauthorized"
            except InvalidAPIKey:
                errors[CONF_API_KEY] = "invalid_api_key"
            except WhisperModelNotFound:
                errors["base"] = "whisper_not_found"
            except UnknownError:
                errors["base"] = "unknown"

        model_schema_key, model_schema_value = await async_model_schema_entry(
            provider, selected_model
        )

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=self.add_suggested_values_to_schema(
                data_schema=vol.Schema(
                    {
                        vol.Required(
                            CONF_NAME, default=f"{provider.name} Whisper"
                        ): cv.string,
                        vol.Optional(CONF_API_KEY): cv.string,
                        model_schema_key: model_schema_value,
                        vol.Optional(
                            CONF_TEMPERATURE, default=DEFAULT_TEMPERATURE
                        ): vol.All(vol.Coerce(float), vol.Range(min=0, max=1)),
                        vol.Optional(CONF_PROMPT): cv.string,
                    }
                ),
                suggested_values={
                    CONF_NAME: entry.data.get(CONF_NAME),
                    CONF_MODEL: selected_model,
                    CONF_TEMPERATURE: entry.options.get(CONF_TEMPERATURE),
                    CONF_PROMPT: entry.options.get(CONF_PROMPT),
                },
            ),
            errors=errors,
        )


class UnknownError(exceptions.HomeAssistantError):
    """Unknown error."""


class UnauthorizedError(exceptions.HomeAssistantError):
    """API key valid but doesn't have the rights to use Whisper."""


class InvalidAPIKey(exceptions.HomeAssistantError):
    """Invalid api_key error."""


class WhisperModelNotFound(exceptions.HomeAssistantError):
    """Whisper Model Not Found error."""
