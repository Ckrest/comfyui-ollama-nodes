"""
Ollama Query Node for ComfyUI
Sends prompts to local Ollama models with full parameter control.
"""

import requests
import json
from typing import Optional, List

# Default Ollama endpoint (socket-activated proxy port)
OLLAMA_HOST = "http://127.0.0.1:11434"

# Fallback models when Ollama is unavailable
DEFAULT_MODELS = ["qwen2.5:0.5b", "llama3.2:latest", "gemma2:latest"]


def get_ollama_models(host: str = OLLAMA_HOST) -> List[str]:
    """Fetch available models from Ollama.

    Returns a list of model names, or fallback defaults if Ollama is unreachable.
    This is called at node registration time to populate the dropdown.
    """
    try:
        response = requests.get(f"{host}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        models = [m.get("name", "unknown") for m in data.get("models", [])]
        return models if models else DEFAULT_MODELS
    except Exception:
        # Return defaults if Ollama isn't running during ComfyUI startup
        return DEFAULT_MODELS


class OllamaQuery:
    """Query an Ollama model with full parameter control.

    Sends a prompt to a local Ollama instance and returns the response.
    All model parameters are exposed for fine-grained control over generation.
    """

    @classmethod
    def IS_CHANGED(cls, model, prompt, system_prompt="", temperature=0.7,
                   top_k=40, top_p=0.9, min_p=0.0, seed=-1, **kwargs):
        """Force re-execution for non-deterministic LLM calls.

        When seed >= 0, Ollama uses deterministic generation, so caching is safe.
        When seed == -1 or None (default), each call should produce different output.
        """
        if seed is not None and seed >= 0:
            # Deterministic mode: allow caching
            return f"{model}:{prompt}:{system_prompt}:{seed}"
        # Non-deterministic: force re-execution
        return float("nan")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Core inputs - model is a dropdown populated from Ollama
                "model": (get_ollama_models(),),
                "prompt": ("STRING", {"multiline": True, "default": "Hello, how are you?"}),
            },
            "optional": {
                # System prompt
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),

                # === Generation Parameters ===
                # Temperature: Higher = more creative, Lower = more deterministic
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),

                # Top-K: Only sample from top K tokens (0 = disabled)
                "top_k": ("INT", {"default": 40, "min": 0, "max": 200}),

                # Top-P (nucleus): Only sample from tokens summing to P probability
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),

                # Min-P: Minimum probability threshold for tokens
                "min_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),

                # Seed: For reproducible generation (-1 = random)
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),

                # Max tokens to generate (num_predict in Ollama)
                "max_tokens": ("INT", {"default": 256, "min": 1, "max": 8192}),

                # === Repetition Control ===
                # Repeat penalty: Higher = less repetition
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 3.0, "step": 0.05}),

                # How far back to look for repetitions
                "repeat_last_n": ("INT", {"default": 64, "min": 0, "max": 2048}),

                # Presence penalty (penalize tokens already in response)
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),

                # Frequency penalty (penalize based on frequency)
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),

                # === Context & Memory ===
                # Context window size (num_ctx)
                "context_size": ("INT", {"default": 4096, "min": 512, "max": 131072}),

                # Number of tokens to keep from context when truncating
                "num_keep": ("INT", {"default": -1, "min": -1, "max": 16384}),

                # === Advanced Sampling ===
                # Mirostat mode (0=disabled, 1=Mirostat, 2=Mirostat 2.0)
                "mirostat": ("INT", {"default": 0, "min": 0, "max": 2}),

                # Mirostat target entropy (tau)
                "mirostat_tau": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.1}),

                # Mirostat learning rate (eta)
                "mirostat_eta": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),

                # TFS-Z: Tail free sampling (1.0 = disabled)
                "tfs_z": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),

                # Typical-P sampling (1.0 = disabled)
                "typical_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),

                # === Performance ===
                # Batch size for prompt processing
                "num_batch": ("INT", {"default": 512, "min": 1, "max": 2048}),

                # Number of threads for generation
                "num_thread": ("INT", {"default": 0, "min": 0, "max": 128}),

                # Number of GPU layers (-1 = auto)
                "num_gpu": ("INT", {"default": -1, "min": -1, "max": 200}),

                # === Stop Sequences ===
                # Comma-separated stop sequences
                "stop_sequences": ("STRING", {"default": ""}),

                # === Connection ===
                # Custom Ollama host URL
                "ollama_host": ("STRING", {"default": OLLAMA_HOST}),

                # Request timeout in seconds
                "timeout": ("INT", {"default": 120, "min": 10, "max": 600}),

                # === Mode ===
                # Raw mode: Skip template formatting
                "raw_mode": ("BOOLEAN", {"default": False}),

                # Whether to stream (usually False for ComfyUI)
                "stream": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("response", "full_json", "token_count")
    FUNCTION = "query"
    CATEGORY = "Ollama"

    def query(
        self,
        model: str,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        min_p: float = 0.0,
        seed: int = -1,
        max_tokens: int = 256,
        repeat_penalty: float = 1.1,
        repeat_last_n: int = 64,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        context_size: int = 4096,
        num_keep: int = -1,
        mirostat: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        tfs_z: float = 1.0,
        typical_p: float = 1.0,
        num_batch: int = 512,
        num_thread: int = 0,
        num_gpu: int = -1,
        stop_sequences: str = "",
        ollama_host: str = OLLAMA_HOST,
        timeout: int = 120,
        raw_mode: bool = False,
        stream: bool = False,
    ):
        """Send query to Ollama and return response."""

        # Build options dict (only include non-default values to keep request clean)
        options = {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "num_predict": max_tokens,
            "repeat_penalty": repeat_penalty,
            "repeat_last_n": repeat_last_n,
            "num_ctx": context_size,
            "mirostat": mirostat,
            "mirostat_tau": mirostat_tau,
            "mirostat_eta": mirostat_eta,
            "tfs_z": tfs_z,
            "typical_p": typical_p,
            "num_batch": num_batch,
        }

        # Add optional parameters only if set
        if min_p > 0:
            options["min_p"] = min_p
        if seed >= 0:
            options["seed"] = seed
        if presence_penalty != 0:
            options["presence_penalty"] = presence_penalty
        if frequency_penalty != 0:
            options["frequency_penalty"] = frequency_penalty
        if num_keep >= 0:
            options["num_keep"] = num_keep
        if num_thread > 0:
            options["num_thread"] = num_thread
        if num_gpu >= 0:
            options["num_gpu"] = num_gpu

        # Build request body
        body = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": options,
        }

        # Add system prompt if provided
        if system_prompt.strip():
            body["system"] = system_prompt.strip()

        # Add stop sequences if provided
        if stop_sequences.strip():
            body["stop"] = [s.strip() for s in stop_sequences.split(",") if s.strip()]

        # Add raw mode if enabled
        if raw_mode:
            body["raw"] = True

        # Make the request
        try:
            response = requests.post(
                f"{ollama_host}/api/generate",
                json=body,
                timeout=timeout
            )
            response.raise_for_status()

            data = response.json()

            # Extract response text
            response_text = data.get("response", "")

            # Get token count from eval_count
            token_count = data.get("eval_count", 0)

            # Full JSON for debugging
            full_json = json.dumps(data, indent=2)

            return (response_text, full_json, token_count)

        except requests.exceptions.ConnectionError:
            error_msg = f"[Error: Cannot connect to Ollama at {ollama_host}]"
            return (error_msg, json.dumps({"error": error_msg}), 0)
        except requests.exceptions.Timeout:
            error_msg = f"[Error: Request timed out after {timeout}s]"
            return (error_msg, json.dumps({"error": error_msg}), 0)
        except requests.exceptions.HTTPError as e:
            error_msg = f"[Error: HTTP {e.response.status_code} - {e.response.text}]"
            return (error_msg, json.dumps({"error": error_msg}), 0)
        except Exception as e:
            error_msg = f"[Error: {str(e)}]"
            return (error_msg, json.dumps({"error": error_msg}), 0)


class OllamaChat:
    """Chat with an Ollama model using message format.

    Uses the /api/chat endpoint for multi-turn conversations.
    Accepts previous messages as JSON for context continuity.
    """

    @classmethod
    def IS_CHANGED(cls, model, user_message, messages_json="[]", system_prompt="",
                   temperature=0.7, top_k=40, top_p=0.9, seed=-1, **kwargs):
        """Force re-execution for non-deterministic LLM calls."""
        if seed is not None and seed >= 0:
            # Deterministic mode: allow caching
            return f"{model}:{user_message}:{messages_json}:{seed}"
        # Non-deterministic: force re-execution
        return float("nan")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Model dropdown populated from Ollama
                "model": (get_ollama_models(),),
                "user_message": ("STRING", {"multiline": True, "default": "Hello!"}),
            },
            "optional": {
                # Previous messages as JSON array
                "messages_json": ("STRING", {"multiline": True, "default": "[]"}),

                # System prompt (added as first message)
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),

                # === Generation Parameters (same as OllamaQuery) ===
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 200}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "max_tokens": ("INT", {"default": 256, "min": 1, "max": 8192}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 3.0, "step": 0.05}),
                "context_size": ("INT", {"default": 4096, "min": 512, "max": 131072}),

                # Connection
                "ollama_host": ("STRING", {"default": OLLAMA_HOST}),
                "timeout": ("INT", {"default": 120, "min": 10, "max": 600}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("response", "updated_messages", "full_json")
    FUNCTION = "chat"
    CATEGORY = "Ollama"

    def chat(
        self,
        model: str,
        user_message: str,
        messages_json: str = "[]",
        system_prompt: str = "",
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        seed: int = -1,
        max_tokens: int = 256,
        repeat_penalty: float = 1.1,
        context_size: int = 4096,
        ollama_host: str = OLLAMA_HOST,
        timeout: int = 120,
    ):
        """Send chat message to Ollama and return response with updated history."""

        # Parse existing messages
        try:
            messages = json.loads(messages_json) if messages_json.strip() else []
        except json.JSONDecodeError:
            messages = []

        # Add system prompt if provided and not already present
        if system_prompt.strip():
            if not messages or messages[0].get("role") != "system":
                messages.insert(0, {"role": "system", "content": system_prompt.strip()})

        # Add user message
        messages.append({"role": "user", "content": user_message})

        # Build options
        options = {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "num_predict": max_tokens,
            "repeat_penalty": repeat_penalty,
            "num_ctx": context_size,
        }

        if seed >= 0:
            options["seed"] = seed

        # Build request
        body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
        }

        try:
            response = requests.post(
                f"{ollama_host}/api/chat",
                json=body,
                timeout=timeout
            )
            response.raise_for_status()

            data = response.json()

            # Extract assistant response
            assistant_message = data.get("message", {})
            response_text = assistant_message.get("content", "")

            # Add assistant response to history
            messages.append({"role": "assistant", "content": response_text})

            # Return response and updated messages
            updated_messages = json.dumps(messages, indent=2)
            full_json = json.dumps(data, indent=2)

            return (response_text, updated_messages, full_json)

        except requests.exceptions.ConnectionError:
            error_msg = f"[Error: Cannot connect to Ollama at {ollama_host}]"
            return (error_msg, messages_json, json.dumps({"error": error_msg}))
        except requests.exceptions.Timeout:
            error_msg = f"[Error: Request timed out after {timeout}s]"
            return (error_msg, messages_json, json.dumps({"error": error_msg}))
        except Exception as e:
            error_msg = f"[Error: {str(e)}]"
            return (error_msg, messages_json, json.dumps({"error": error_msg}))


class OllamaModelList:
    """List available Ollama models.

    Fetches the list of models installed on the local Ollama instance.
    Useful for debugging and checking available models.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "ollama_host": ("STRING", {"default": OLLAMA_HOST}),
                "trigger": ("*", {}),  # Optional trigger to force refresh
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("models_list", "models_json")
    FUNCTION = "list_models"
    CATEGORY = "Ollama"

    def list_models(self, ollama_host: str = OLLAMA_HOST, trigger=None):
        """Fetch list of available models from Ollama."""

        try:
            response = requests.get(
                f"{ollama_host}/api/tags",
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            models = data.get("models", [])

            # Create simple list of model names
            model_names = [m.get("name", "unknown") for m in models]
            models_list = "\n".join(model_names)

            # Full JSON for details
            models_json = json.dumps(data, indent=2)

            return (models_list, models_json)

        except requests.exceptions.ConnectionError:
            error_msg = f"Cannot connect to Ollama at {ollama_host}"
            return (f"[Error: {error_msg}]", json.dumps({"error": error_msg}))
        except Exception as e:
            return (f"[Error: {str(e)}]", json.dumps({"error": str(e)}))


# Node registration
NODE_CLASS_MAPPINGS = {
    "OllamaQuery": OllamaQuery,
    "OllamaChat": OllamaChat,
    "OllamaModelList": OllamaModelList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaQuery": "Ollama Query 🦙",
    "OllamaChat": "Ollama Chat 💬",
    "OllamaModelList": "Ollama Model List 📋",
}
