import litellm
from litellm.integrations.custom_logger import CustomLogger

class TokenizeTranslator(CustomLogger):
    def __init__(self):
        super().__init__()

    def log_pre_api_call(self, model, messages, kwargs):
        print(f"Pre-API log Call: {kwargs}")

    # 1. TRANSLATE REQUEST: vLLM -> Watsonx
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        print(f"Pre-API call")
        print(data)
        # Only apply this transformation if it looks like a vLLM tokenize request
        if "prompt" in data and call_type == "pass_through_endpoint": # LiteLLM passes this type for passthrough routes
            # Extract the vLLM string
            prompt_string = data.pop("prompt")
            data.pop("model", None)
            
            # Reshape exactly to what IBM Watsonx expects
            data["input"] = prompt_string
            data["model_id"] = data.get("model", "meta-llama/llama-3-3-70b-instruct")
            data["project_id"] = litellm.os.environ.get("WATSONX_PROJECT_ID")
            data["parameters"] = {
                "return_tokens": "true"
            }
            
        return data

    # 2. TRANSLATE RESPONSE: Watsonx -> vLLM
    async def async_post_call_success_hook(self, data, user_api_key_dict, response):

        # SAFEGUARD: Ensure this was specifically a pass-through request
        is_pass_through = data.get("call_type") == "pass_through"
        is_tokenize_route = "/tokenize" in data.get("url", "")

        if not (is_pass_through and is_tokenize_route):
            return response # Immediately let normal chat traffic pass untouched

        # Check if this is the specific IBM tokenization response format
        if isinstance(response, dict) and "result" in response:
            try:
                # Extract IBM's token array and count directly from the 'result' object
                tokens = response["result"]["tokens"]
                token_count = response["result"].get("token_count", len(tokens))

                # Overwrite the response dict with vLLM format
                response.clear()
                response["tokens"] = tokens
                response["count"] = token_count
            except (KeyError, TypeError):
                # If the schema still doesn't match perfectly, fail gracefully
                pass

        return response

# Instantiate it so the config can import it
translator_instance = TokenizeTranslator()
