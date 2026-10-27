from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Final, cast  # noqa: TID251  # proxy globals untyped

from fastapi import Request
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm._logging import verbose_router_logger
from litellm.exceptions import BadRequestError
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.proxy.common_utils.http_parsing_utils import get_form_data

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.router import DeploymentTypedDict, LiteLLMParamsTypedDict


class WatsonxPassThroughCredentials(TypedDict):
    api_key: ReadOnly[NotRequired[str | None]]
    api_base: ReadOnly[NotRequired[str | None]]
    project_id: ReadOnly[NotRequired[str | None]]
    space_id: ReadOnly[NotRequired[str | None]]
    model: ReadOnly[NotRequired[str | None]]


_SPECIAL_MODEL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "all-team-models",
        "all-proxy-models",
        "no-default-models",
    }
)


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _has_wildcard_or_special_token(models: Sequence[str]) -> bool:
    return any("*" in m or m in _SPECIAL_MODEL_NAMES for m in models)


def _needs_dynamic_model_resolution(
    user_api_key_dict: "UserAPIKeyAuth",
    model_access_groups: Mapping[str, Sequence[str]],
) -> bool:
    if bool(user_api_key_dict.access_group_ids) or bool(getattr(user_api_key_dict, "config", None)):
        return True
    models: Final[Sequence[str]] = cast(Sequence[str], user_api_key_dict.models) or ()  # cast-ok: key models
    team_models: Final[Sequence[str]] = cast(Sequence[str], user_api_key_dict.team_models) or ()  # cast-ok: team models
    if _has_wildcard_or_special_token(models) or _has_wildcard_or_special_token(team_models):
        return True
    return bool(model_access_groups and any(m in model_access_groups for m in (*models, *team_models)))


class WatsonxPassthroughHandler:
    @staticmethod
    async def request_body(request: Request) -> Mapping[str, object]:
        if request.method != "POST":
            return {}
        if "multipart/form-data" in request.headers.get("content-type", ""):
            return await get_form_data(request)
        return cast(Mapping[str, object], await request.json())  # cast-ok: json payload

    @staticmethod
    def extract_param(
        request_body: Mapping[str, object],
        query_params: Mapping[str, str],
        name: str,
        alias: str | None = None,
    ) -> str | None:
        return (
            _str_or_none(request_body.get(name))
            or _str_or_none(query_params.get(name))
            or (_str_or_none(request_body.get(alias)) or _str_or_none(query_params.get(alias)) if alias else None)
        )

    @classmethod
    async def allowed_deployments(
        cls,
        user_api_key_dict: "UserAPIKeyAuth",
        requested_model: str | None,
    ) -> Sequence["DeploymentTypedDict"]:
        from fastapi import status

        from litellm.proxy._types import ModelAccessDeniedProxyException, ProxyErrorTypes
        from litellm.proxy.auth.auth_checks import model_access_denied_client_message
        from litellm.proxy.proxy_server import (
            general_settings,  # pyright: ignore[reportUnknownVariableType]  # untyped proxy global
            llm_router,
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
            user_model,  # pyright: ignore[reportUnknownVariableType]  # untyped proxy global
        )
        from litellm.proxy.utils import (
            get_available_models_for_user,  # pyright: ignore[reportUnknownVariableType]  # untyped settings parameter
        )

        if llm_router is None:
            return ()
        all_deployments: Final = tuple(
            d for d in (llm_router.get_model_list() or ()) if d["litellm_params"].get("use_in_pass_through") is True
        )
        if not all_deployments:
            return ()

        models: Final[Sequence[str]] = cast(Sequence[str], user_api_key_dict.models) or ()  # cast-ok: key models
        team_models: Final[Sequence[str]] = (
            cast(Sequence[str], user_api_key_dict.team_models) or ()  # cast-ok: team models
        )
        has_model_restrictions: Final = bool(models) or bool(team_models) or bool(user_api_key_dict.access_group_ids)
        if not has_model_restrictions:
            return all_deployments

        model_access_groups: Final = llm_router.get_model_access_groups() if llm_router else {}
        available_models: Final = (
            frozenset(
                await get_available_models_for_user(
                    user_api_key_dict=user_api_key_dict,
                    llm_router=llm_router,
                    general_settings=cast(dict[str, object], general_settings),  # cast-ok: proxy server settings
                    user_model=cast(str | None, user_model),  # cast-ok: proxy server untyped global
                    prisma_client=prisma_client,
                    proxy_logging_obj=proxy_logging_obj,
                    user_api_key_cache=user_api_key_cache,
                )
            )
            if _needs_dynamic_model_resolution(user_api_key_dict, model_access_groups)
            else frozenset(models or team_models)
        )
        allowed_deployments: Final = tuple(
            d
            for d in all_deployments
            if d["model_name"] in available_models
            or (isinstance(m := d["litellm_params"].get("model"), str) and m in available_models)
        )
        if requested_model is None:
            return allowed_deployments
        model_is_accessible: Final = (
            requested_model in available_models
            or f"watsonx/{requested_model}" in available_models
            or any(
                cls.deployment_model_matches(creds, requested_model)
                for d in allowed_deployments
                if (creds := cls.resolve_credentials_from_deployment(d["litellm_params"])) is not None
            )
        )
        if not model_is_accessible:
            raise ModelAccessDeniedProxyException(
                message=model_access_denied_client_message(model=requested_model),
                internal_message=f"Key not allowed to access model {requested_model}",
                type=ProxyErrorTypes.key_model_access_denied,
                param="model",
                code=status.HTTP_403_FORBIDDEN,
            )
        return allowed_deployments

    @classmethod
    def resolve_credentials_from_router_deployments(
        cls,
        model_id: str | None = None,
        project_id: str | None = None,
        space_id: str | None = None,
        deployments: "Sequence[DeploymentTypedDict] | None" = None,
    ) -> "WatsonxPassThroughCredentials | None":
        candidates: Final = tuple(
            creds
            for deployment in (deployments or ())
            if (creds := cls.resolve_credentials_from_deployment(deployment["litellm_params"])) is not None
        )
        if not candidates:
            return None

        first_candidate: Final = candidates[0]

        def model_ok(creds: WatsonxPassThroughCredentials) -> bool:
            return bool(model_id and cls.deployment_model_matches(creds, model_id))

        def project_ok(creds: WatsonxPassThroughCredentials) -> bool:
            return bool(project_id) and creds.get("project_id") == project_id

        def space_ok(creds: WatsonxPassThroughCredentials) -> bool:
            return bool(space_id) and creds.get("space_id") == space_id

        selection_rules: Final[tuple[Callable[[WatsonxPassThroughCredentials], bool], ...]] = (
            lambda creds: model_ok(creds) and project_ok(creds),
            lambda creds: model_ok(creds) and space_ok(creds),
            model_ok,
            project_ok,
            space_ok,
        )

        for rule in selection_rules:
            if (matched := next(filter(rule, candidates), None)) is not None:
                return matched

        if len(candidates) > 1:
            verbose_router_logger.warning(
                "Multiple watsonx use_in_pass_through deployments found but no "
                "model_id/project_id/space_id supplied to disambiguate — using the first. "
                "Set project_id or space_id on each deployment's litellm_params and pass "
                "project_id or space_id in the request body to route to the correct credentials."
            )
        return first_candidate

    @staticmethod
    def deployment_model_matches(creds: "WatsonxPassThroughCredentials", model_id: str) -> bool:
        stored: Final = creds.get("model")
        if stored is None:
            return False
        return stored == model_id or str(stored).endswith(f"/{model_id}")

    @classmethod
    def resolve_credentials_from_deployment(
        cls, litellm_params: "LiteLLMParamsTypedDict"
    ) -> "WatsonxPassThroughCredentials | None":
        model: Final = litellm_params.get("model")
        if model is None:
            return None
        try:
            _, provider, _, _ = get_llm_provider(
                model=model,
                custom_llm_provider=litellm_params.get("custom_llm_provider"),
            )
        except BadRequestError:
            return None
        if provider != "watsonx":
            return None
        credential_name: Final = litellm_params.get("litellm_credential_name")
        credential_values: Final[Mapping[str, object]] = (
            CredentialAccessor.get_credential_values(credential_name) if credential_name is not None else {}
        )
        credential_api_key: Final = credential_values.get("api_key")
        credential_api_base: Final = credential_values.get("api_base")
        api_key: Final = (credential_api_key if isinstance(credential_api_key, str) else None) or litellm_params.get(
            "api_key"
        )
        api_base: Final = (credential_api_base if isinstance(credential_api_base, str) else None) or litellm_params.get(
            "api_base"
        )
        return WatsonxPassThroughCredentials(
            api_key=api_key,
            api_base=api_base,
            project_id=litellm_params.get("project_id"),
            space_id=litellm_params.get("space_id"),
            model=model,
        )
