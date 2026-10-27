"""
Unit tests for WatsonxPassthroughHandler credential resolution and model matching.
"""

from unittest.mock import patch

import litellm
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.watsonx_passthrough_handler import (
    WatsonxPassthroughHandler,
)


def _wx_deployment(model: str, **litellm_params) -> dict:
    return {
        "model_name": model.split("/", 1)[-1],
        "litellm_params": {"model": model, "use_in_pass_through": True, **litellm_params},
    }


def _wx_unflagged_deployment(model: str, **litellm_params) -> dict:
    return {
        "model_name": model.split("/", 1)[-1],
        "litellm_params": {"model": model, **litellm_params},
    }


class TestDeploymentModelMatches:
    """Tests for WatsonxPassthroughHandler.deployment_model_matches."""

    def test_exact_match(self):
        creds = {"model": "ibm/granite-13b-chat-v2"}
        assert WatsonxPassthroughHandler.deployment_model_matches(creds, "ibm/granite-13b-chat-v2") is True

    def test_suffix_match(self):
        creds = {"model": "watsonx/ibm/granite-13b-chat-v2"}
        assert WatsonxPassthroughHandler.deployment_model_matches(creds, "ibm/granite-13b-chat-v2") is True

    def test_no_match_different_model(self):
        creds = {"model": "watsonx/ibm/granite-8b-code"}
        assert WatsonxPassthroughHandler.deployment_model_matches(creds, "ibm/granite-13b-chat-v2") is False

    def test_no_match_when_model_none(self):
        creds = {"model": None}
        assert WatsonxPassthroughHandler.deployment_model_matches(creds, "ibm/granite-13b-chat-v2") is False

    def test_no_match_when_model_key_absent(self):
        creds = {}
        assert WatsonxPassthroughHandler.deployment_model_matches(creds, "ibm/granite-13b-chat-v2") is False

    def test_partial_suffix_does_not_match(self):
        creds = {"model": "watsonx/ibm/granite-13b-chat-v2"}
        assert WatsonxPassthroughHandler.deployment_model_matches(creds, "granite-13b-chat-v2") is True

    def test_wrong_suffix_does_not_match(self):
        creds = {"model": "watsonx/ibm/granite-13b-chat-v2"}
        assert WatsonxPassthroughHandler.deployment_model_matches(creds, "ibm/granite-8b") is False


class TestResolveCredentialsFromDeployment:
    """Tests for WatsonxPassthroughHandler.resolve_credentials_from_deployment."""

    def test_returns_none_for_non_watsonx_provider(self):
        deployment = _wx_deployment("openai/gpt-4o", api_key="sk-openai")
        result = WatsonxPassthroughHandler.resolve_credentials_from_deployment(
            deployment["litellm_params"]
        )
        assert result is None

    def test_returns_none_when_model_missing(self):
        params = {"use_in_pass_through": True}
        result = WatsonxPassthroughHandler.resolve_credentials_from_deployment(params)
        assert result is None

    def test_returns_none_when_get_llm_provider_raises_bad_request(self):
        from litellm.exceptions import BadRequestError

        params = {"use_in_pass_through": True, "model": "completely-unknown/some-model"}
        with patch(
            "litellm.proxy.pass_through_endpoints.llm_provider_handlers.watsonx_passthrough_handler.get_llm_provider",
            side_effect=BadRequestError(
                message="LLM Provider NOT provided.", model="completely-unknown/some-model", llm_provider=""
            ),
        ):
            result = WatsonxPassthroughHandler.resolve_credentials_from_deployment(params)
        assert result is None

    def test_inline_api_key_and_api_base(self):
        deployment = _wx_deployment(
            "watsonx/ibm/granite-13b-chat-v2",
            api_key="ibmkey-123",
            api_base="https://us-south.ml.cloud.ibm.com",
            project_id="proj-abc",
        )
        result = WatsonxPassthroughHandler.resolve_credentials_from_deployment(
            deployment["litellm_params"]
        )
        assert result is not None
        assert result.get("api_key") == "ibmkey-123"
        assert result.get("api_base") == "https://us-south.ml.cloud.ibm.com"
        assert result.get("project_id") == "proj-abc"
        assert result.get("space_id") is None
        assert result.get("model") == "watsonx/ibm/granite-13b-chat-v2"

    def test_space_id_is_read(self):
        deployment = _wx_deployment(
            "watsonx/ibm/granite-13b-chat-v2",
            api_key="ibmkey-123",
            api_base="https://us-south.ml.cloud.ibm.com",
            space_id="space-xyz",
        )
        result = WatsonxPassthroughHandler.resolve_credentials_from_deployment(
            deployment["litellm_params"]
        )
        assert result is not None
        assert result.get("space_id") == "space-xyz"
        assert result.get("project_id") is None

    def test_api_key_none_when_omitted(self):
        deployment = _wx_deployment(
            "watsonx/ibm/granite-13b-chat-v2",
            api_base="https://us-south.ml.cloud.ibm.com",
        )
        result = WatsonxPassthroughHandler.resolve_credentials_from_deployment(
            deployment["litellm_params"]
        )
        assert result is not None
        assert result.get("api_key") is None
        assert result.get("api_base") == "https://us-south.ml.cloud.ibm.com"

    def test_api_base_none_when_omitted(self):
        deployment = _wx_deployment(
            "watsonx/ibm/granite-13b-chat-v2",
            api_key="ibmkey-123",
        )
        result = WatsonxPassthroughHandler.resolve_credentials_from_deployment(
            deployment["litellm_params"]
        )
        assert result is not None
        assert result.get("api_key") == "ibmkey-123"
        assert result.get("api_base") is None

    def test_named_credential_overrides_inline(self, monkeypatch):
        monkeypatch.setattr(litellm, "credential_list", [])
        from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
        from litellm.types.utils import CredentialItem

        CredentialAccessor.upsert_credentials([
            CredentialItem(
                credential_name="wx-cred",
                credential_values={
                    "api_key": "cred-key-999",
                    "api_base": "https://eu-de.ml.cloud.ibm.com",
                },
                credential_info={},
            )
        ])
        deployment = _wx_deployment(
            "watsonx/ibm/granite-13b-chat-v2",
            litellm_credential_name="wx-cred",
            api_key="inline-key-should-not-win",
            api_base="https://us-south.ml.cloud.ibm.com",
        )
        result = WatsonxPassthroughHandler.resolve_credentials_from_deployment(
            deployment["litellm_params"]
        )
        assert result is not None
        assert result.get("api_key") == "cred-key-999"
        assert result.get("api_base") == "https://eu-de.ml.cloud.ibm.com"

    def test_named_credential_loaded_after_deployment_still_resolves(self, monkeypatch):
        monkeypatch.setattr(litellm, "credential_list", [])
        from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
        from litellm.types.utils import CredentialItem

        deployment = _wx_deployment(
            "watsonx/ibm/granite-13b-chat-v2",
            litellm_credential_name="wx-late",
        )
        result = WatsonxPassthroughHandler.resolve_credentials_from_deployment(
            deployment["litellm_params"]
        )
        assert result is not None
        assert result.get("api_key") is None

        CredentialAccessor.upsert_credentials([
            CredentialItem(
                credential_name="wx-late",
                credential_values={"api_key": "late-key", "api_base": "https://us-south.ml.cloud.ibm.com"},
                credential_info={},
            )
        ])

        result2 = WatsonxPassthroughHandler.resolve_credentials_from_deployment(
            deployment["litellm_params"]
        )
        assert result2 is not None
        assert result2.get("api_key") == "late-key"


class TestResolveCredentialsFromRouterDeployments:
    """Tests for WatsonxPassthroughHandler.resolve_credentials_from_router_deployments."""

    def _deployments(self, *models_and_params):
        return [_wx_deployment(m, **p) for m, p in models_and_params]

    def test_returns_none_with_no_deployments(self):
        result = WatsonxPassthroughHandler.resolve_credentials_from_router_deployments(
            deployments=[]
        )
        assert result is None

    def test_returns_none_with_no_eligible_deployments(self):
        deployments = [_wx_deployment("openai/gpt-4o", api_key="k")]
        result = WatsonxPassthroughHandler.resolve_credentials_from_router_deployments(
            deployments=deployments
        )
        assert result is None

    def test_single_deployment_returned_as_fallback(self):
        deployments = self._deployments(
            ("watsonx/ibm/granite-13b-chat-v2", {"api_key": "k1", "api_base": "https://us-south.ml.cloud.ibm.com"}),
        )
        result = WatsonxPassthroughHandler.resolve_credentials_from_router_deployments(
            deployments=deployments
        )
        assert result is not None
        assert result.get("api_key") == "k1"

    def test_exact_match_model_and_project_id(self):
        deployments = self._deployments(
            ("watsonx/ibm/granite-13b-chat-v2", {"api_key": "k-us", "api_base": "https://us-south.ml.cloud.ibm.com", "project_id": "proj-us"}),
            ("watsonx/ibm/granite-13b-chat-v2", {"api_key": "k-eu", "api_base": "https://eu-de.ml.cloud.ibm.com", "project_id": "proj-eu"}),
        )
        result = WatsonxPassthroughHandler.resolve_credentials_from_router_deployments(
            model_id="ibm/granite-13b-chat-v2",
            project_id="proj-eu",
            deployments=deployments,
        )
        assert result is not None
        assert result.get("api_key") == "k-eu"
        assert result.get("api_base") == "https://eu-de.ml.cloud.ibm.com"

    def test_exact_match_model_and_space_id(self):
        deployments = self._deployments(
            ("watsonx/ibm/granite-13b-chat-v2", {"api_key": "k-studio", "api_base": "https://us-south.ml.cloud.ibm.com", "project_id": "proj-studio"}),
            ("watsonx/ibm/granite-13b-chat-v2", {"api_key": "k-deploy", "api_base": "https://us-south.ml.cloud.ibm.com", "space_id": "space-prod"}),
        )
        result = WatsonxPassthroughHandler.resolve_credentials_from_router_deployments(
            model_id="ibm/granite-13b-chat-v2",
            space_id="space-prod",
            deployments=deployments,
        )
        assert result is not None
        assert result.get("api_key") == "k-deploy"

    def test_partial_match_model_only(self):
        deployments = self._deployments(
            ("watsonx/ibm/granite-8b-code", {"api_key": "k-code", "api_base": "https://us-south.ml.cloud.ibm.com"}),
            ("watsonx/ibm/granite-13b-chat-v2", {"api_key": "k-chat", "api_base": "https://us-south.ml.cloud.ibm.com"}),
        )
        result = WatsonxPassthroughHandler.resolve_credentials_from_router_deployments(
            model_id="ibm/granite-13b-chat-v2",
            deployments=deployments,
        )
        assert result is not None
        assert result.get("api_key") == "k-chat"

    def test_fallback_first_when_no_hints(self):
        deployments = self._deployments(
            ("watsonx/ibm/granite-8b-code", {"api_key": "k-first", "api_base": "https://us-south.ml.cloud.ibm.com"}),
            ("watsonx/ibm/granite-13b-chat-v2", {"api_key": "k-second", "api_base": "https://us-south.ml.cloud.ibm.com"}),
        )
        result = WatsonxPassthroughHandler.resolve_credentials_from_router_deployments(
            deployments=deployments,
        )
        assert result is not None
        assert result.get("api_key") == "k-first"

    def test_non_watsonx_deployments_ignored(self):
        deployments = [
            _wx_deployment("openai/gpt-4o", api_key="openai-key"),
            _wx_deployment("watsonx/ibm/granite-13b-chat-v2", api_key="wx-key", api_base="https://us-south.ml.cloud.ibm.com"),
        ]
        result = WatsonxPassthroughHandler.resolve_credentials_from_router_deployments(
            deployments=deployments,
        )
        assert result is not None
        assert result.get("api_key") == "wx-key"

    def test_invalid_deployments_ignored(self):
        deployments = [
            _wx_deployment("openai/gpt-4o", api_key="should-be-ignored"),
            _wx_deployment("watsonx/ibm/granite-8b-code", api_key="eligible", api_base="https://us-south.ml.cloud.ibm.com"),
        ]
        result = WatsonxPassthroughHandler.resolve_credentials_from_router_deployments(
            deployments=deployments,
        )
        assert result is not None
        assert result.get("api_key") == "eligible"

    def test_project_id_mismatch_falls_through_to_model_match(self):
        deployments = self._deployments(
            ("watsonx/ibm/granite-13b-chat-v2", {"api_key": "k-other", "api_base": "https://us-south.ml.cloud.ibm.com", "project_id": "proj-other"}),
            ("watsonx/ibm/granite-13b-chat-v2", {"api_key": "k-matched", "api_base": "https://us-south.ml.cloud.ibm.com", "project_id": "proj-target"}),
        )
        result = WatsonxPassthroughHandler.resolve_credentials_from_router_deployments(
            model_id="ibm/granite-13b-chat-v2",
            project_id="proj-target",
            deployments=deployments,
        )
        assert result is not None
        assert result.get("api_key") == "k-matched"

    def test_match_project_id_only_without_model_id(self):
        deployments = self._deployments(
            ("watsonx/ibm/granite-13b-chat-v2", {"api_key": "k-proj-a", "api_base": "https://us-south.ml.cloud.ibm.com", "project_id": "proj-a"}),
            ("watsonx/ibm/granite-8b-code", {"api_key": "k-proj-b", "api_base": "https://us-south.ml.cloud.ibm.com", "project_id": "proj-b"}),
        )
        result = WatsonxPassthroughHandler.resolve_credentials_from_router_deployments(
            project_id="proj-b",
            deployments=deployments,
        )
        assert result is not None
        assert result.get("api_key") == "k-proj-b"

    def test_match_space_id_only_without_model_id(self):
        deployments = self._deployments(
            ("watsonx/ibm/granite-13b-chat-v2", {"api_key": "k-space-a", "api_base": "https://us-south.ml.cloud.ibm.com", "space_id": "space-a"}),
            ("watsonx/ibm/granite-8b-code", {"api_key": "k-space-b", "api_base": "https://us-south.ml.cloud.ibm.com", "space_id": "space-b"}),
        )
        result = WatsonxPassthroughHandler.resolve_credentials_from_router_deployments(
            space_id="space-b",
            deployments=deployments,
        )
        assert result is not None
        assert result.get("api_key") == "k-space-b"


class TestWatsonxPassthroughHandlerHelpers:
    """Tests for request_body and extract_param helpers."""

    async def test_request_body_get_returns_empty(self):
        from unittest.mock import MagicMock

        request = MagicMock()
        request.method = "GET"
        result = await WatsonxPassthroughHandler.request_body(request)
        assert result == {}

    async def test_request_body_json(self):
        from unittest.mock import AsyncMock, MagicMock

        request = MagicMock()
        request.method = "POST"
        request.headers = {"content-type": "application/json"}
        request.json = AsyncMock(return_value={"model_id": "ibm/granite-13b"})
        result = await WatsonxPassthroughHandler.request_body(request)
        assert result == {"model_id": "ibm/granite-13b"}

    def test_extract_param_body_primary(self):
        body = {"model_id": "model-1"}
        query = {}
        assert WatsonxPassthroughHandler.extract_param(body, query, "model_id", "model") == "model-1"

    def test_extract_param_body_alias(self):
        body = {"model": "model-alias"}
        query = {}
        assert WatsonxPassthroughHandler.extract_param(body, query, "model_id", "model") == "model-alias"

    def test_extract_param_query_primary(self):
        body = {}
        query = {"model_id": "model-query"}
        assert WatsonxPassthroughHandler.extract_param(body, query, "model_id", "model") == "model-query"

    def test_extract_param_query_alias(self):
        body = {}
        query = {"model": "model-query-alias"}
        assert WatsonxPassthroughHandler.extract_param(body, query, "model_id", "model") == "model-query-alias"

    def test_extract_param_none_when_absent(self):
        body = {}
        query = {}
        assert WatsonxPassthroughHandler.extract_param(body, query, "model_id", "model") is None


class TestAllowedDeployments:
    """Tests for WatsonxPassthroughHandler.allowed_deployments fast-path and slow-path."""

    async def test_fast_path_explicit_key_models(self):
        from unittest.mock import AsyncMock, MagicMock, patch
        from litellm.proxy._types import UserAPIKeyAuth

        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            _wx_deployment("watsonx/ibm/granite-13b-chat-v2", api_key="k1"),
            _wx_deployment("watsonx/ibm/granite-8b-code", api_key="k2"),
        ]
        mock_router.get_model_access_groups.return_value = {}

        user_auth = UserAPIKeyAuth(models=["ibm/granite-13b-chat-v2"])

        with (
            patch("litellm.proxy.proxy_server.llm_router", mock_router),
            patch(
                "litellm.proxy.utils.get_available_models_for_user",
                new_callable=AsyncMock,
            ) as mock_get_models,
        ):
            deployments = await WatsonxPassthroughHandler.allowed_deployments(
                user_api_key_dict=user_auth,
                requested_model="ibm/granite-13b-chat-v2",
            )
            assert len(deployments) == 1
            assert deployments[0]["model_name"] == "ibm/granite-13b-chat-v2"
            mock_get_models.assert_not_called()

    async def test_fast_path_explicit_team_models(self):
        from unittest.mock import AsyncMock, MagicMock, patch
        from litellm.proxy._types import UserAPIKeyAuth

        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            _wx_deployment("watsonx/ibm/granite-13b-chat-v2", api_key="k1"),
        ]
        mock_router.get_model_access_groups.return_value = {}

        user_auth = UserAPIKeyAuth(models=[], team_models=["ibm/granite-13b-chat-v2"])

        with (
            patch("litellm.proxy.proxy_server.llm_router", mock_router),
            patch(
                "litellm.proxy.utils.get_available_models_for_user",
                new_callable=AsyncMock,
            ) as mock_get_models,
        ):
            deployments = await WatsonxPassthroughHandler.allowed_deployments(
                user_api_key_dict=user_auth,
                requested_model="ibm/granite-13b-chat-v2",
            )
            assert len(deployments) == 1
            mock_get_models.assert_not_called()

    async def test_slow_path_wildcard_model(self):
        from unittest.mock import AsyncMock, MagicMock, patch
        from litellm.proxy._types import UserAPIKeyAuth

        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            _wx_deployment("watsonx/ibm/granite-13b-chat-v2", api_key="k1"),
        ]
        mock_router.get_model_access_groups.return_value = {}

        user_auth = UserAPIKeyAuth(models=["watsonx/*"])

        with (
            patch("litellm.proxy.proxy_server.llm_router", mock_router),
            patch(
                "litellm.proxy.utils.get_available_models_for_user",
                new_callable=AsyncMock,
                return_value=["watsonx/ibm/granite-13b-chat-v2"],
            ) as mock_get_models,
        ):
            deployments = await WatsonxPassthroughHandler.allowed_deployments(
                user_api_key_dict=user_auth,
                requested_model="ibm/granite-13b-chat-v2",
            )
            assert len(deployments) == 1
            mock_get_models.assert_called_once()

    async def test_slow_path_special_token(self):
        from unittest.mock import AsyncMock, MagicMock, patch
        from litellm.proxy._types import UserAPIKeyAuth

        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            _wx_deployment("watsonx/ibm/granite-13b-chat-v2", api_key="k1"),
        ]
        mock_router.get_model_access_groups.return_value = {}

        user_auth = UserAPIKeyAuth(models=["all-team-models"])

        with (
            patch("litellm.proxy.proxy_server.llm_router", mock_router),
            patch(
                "litellm.proxy.utils.get_available_models_for_user",
                new_callable=AsyncMock,
                return_value=["watsonx/ibm/granite-13b-chat-v2"],
            ) as mock_get_models,
        ):
            deployments = await WatsonxPassthroughHandler.allowed_deployments(
                user_api_key_dict=user_auth,
                requested_model="ibm/granite-13b-chat-v2",
            )
            assert len(deployments) == 1
            mock_get_models.assert_called_once()

    async def test_slow_path_access_group_ids(self):
        from unittest.mock import AsyncMock, MagicMock, patch
        from litellm.proxy._types import UserAPIKeyAuth

        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            _wx_deployment("watsonx/ibm/granite-13b-chat-v2", api_key="k1"),
        ]
        mock_router.get_model_access_groups.return_value = {}

        user_auth = UserAPIKeyAuth(models=[], access_group_ids=["group-1"])

        with (
            patch("litellm.proxy.proxy_server.llm_router", mock_router),
            patch(
                "litellm.proxy.utils.get_available_models_for_user",
                new_callable=AsyncMock,
                return_value=["watsonx/ibm/granite-13b-chat-v2"],
            ) as mock_get_models,
        ):
            deployments = await WatsonxPassthroughHandler.allowed_deployments(
                user_api_key_dict=user_auth,
                requested_model="ibm/granite-13b-chat-v2",
            )
            assert len(deployments) == 1
            mock_get_models.assert_called_once()

    async def test_unrestricted_key_skips_model_check(self):
        from unittest.mock import AsyncMock, MagicMock, patch
        from litellm.proxy._types import UserAPIKeyAuth

        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            _wx_deployment("watsonx/ibm/granite-13b-chat-v2", api_key="k1"),
            _wx_deployment("watsonx/ibm/granite-8b-code", api_key="k2"),
        ]

        user_auth = UserAPIKeyAuth(models=[], team_models=[], access_group_ids=None)

        with (
            patch("litellm.proxy.proxy_server.llm_router", mock_router),
            patch(
                "litellm.proxy.utils.get_available_models_for_user",
                new_callable=AsyncMock,
            ) as mock_get_models,
        ):
            deployments = await WatsonxPassthroughHandler.allowed_deployments(
                user_api_key_dict=user_auth,
                requested_model="ibm/granite-13b-chat-v2",
            )
            assert len(deployments) == 2
            mock_get_models.assert_not_called()

    async def test_access_denied_raises_exception(self):
        import pytest
        from unittest.mock import MagicMock, patch
        from litellm.proxy._types import ModelAccessDeniedProxyException, UserAPIKeyAuth

        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            _wx_deployment("watsonx/ibm/granite-13b-chat-v2", api_key="k1"),
        ]
        mock_router.get_model_access_groups.return_value = {}

        user_auth = UserAPIKeyAuth(models=["ibm/granite-13b-chat-v2"])

        with (
            patch("litellm.proxy.proxy_server.llm_router", mock_router),
            pytest.raises(ModelAccessDeniedProxyException) as exc_info,
        ):
            await WatsonxPassthroughHandler.allowed_deployments(
                user_api_key_dict=user_auth,
                requested_model="ibm/granite-8b-code",
            )
        assert str(exc_info.value.code) == "403"
