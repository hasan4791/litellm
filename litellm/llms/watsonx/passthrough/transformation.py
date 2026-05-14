from typing import TYPE_CHECKING, List, Optional, Tuple

import httpx

from litellm.llms.base_llm.passthrough.transformation import BasePassthroughConfig
from litellm.llms.watsonx.common_utils import generate_iam_token
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.router import GenericLiteLLMParams

if TYPE_CHECKING:
    from httpx import URL


class WatsonxPassthroughConfig(BasePassthroughConfig):
    """
    Watsonx-specific passthrough configuration.
    Handles:
    - IAM token generation and caching
    - Version parameter injection
    - Project ID management
    """

    def is_streaming_request(self, endpoint: str, request_data: dict) -> bool:
        """Check if request should be streamed"""
        return request_data.get("stream", False)

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        endpoint: str,
        request_query_params: Optional[dict],
        litellm_params: dict,
    ) -> Tuple["URL", str]:
        """
        Construct complete Watsonx URL with version parameter.
        
        This ensures the version parameter is ALWAYS included in the URL,
        solving the query parameter issue.
        """
        print("chasan")
        base_target_url = self.get_api_base(api_base)

        if base_target_url is None:
            raise Exception("Watsonx API base not found")

        # Ensure version parameter is always present
        if request_query_params is None:
            request_query_params = {}
        
        # Add version if not already present (default to 2023-05-02)
        if "version" not in request_query_params:
            request_query_params["version"] = "2023-05-02"

        # Use the format_url helper to construct URL with query params
        complete_url = self.format_url(
            endpoint=endpoint,
            base_target_url=base_target_url,
            request_query_params=request_query_params,
        )

        return (complete_url, base_target_url)

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Validate Watsonx credentials and inject IAM token.
        
        This automatically generates and caches IAM tokens,
        solving the token management issue.
        """
        # Get API key from params or environment
        if api_key is None:
            api_key = self.get_api_key()

        if api_key is None:
            raise ValueError("Watsonx API key is required")

        # Generate IAM token (cached internally by generate_iam_token)
        iam_token = generate_iam_token(api_key=api_key)

        # Set Authorization header with Bearer token
        headers["Authorization"] = f"Bearer {iam_token}"
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json"

        return headers

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> Optional[str]:
        """Get Watsonx API base URL"""
        return (
            api_base
            or get_secret_str("WATSONX_URL")
            or get_secret_str("WATSONX_BASE_URL")
            or "https://us-south.ml.cloud.ibm.com"
        )

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        """Get Watsonx API key"""
        return (
            api_key
            or get_secret_str("WATSONX_API_KEY")
            or get_secret_str("WATSONX_APIKEY")
            or get_secret_str("WX_API_KEY")
        )

    @staticmethod
    def get_base_model(model: str) -> Optional[str]:
        """Return base model name"""
        return model

    def get_models(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None
    ) -> List[str]:
        """Get available models (optional implementation)"""
        return super().get_models(api_key, api_base)
