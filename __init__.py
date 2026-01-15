"""
ComfyUI Ollama Nodes - Local LLM integration via Ollama

Nodes:
- OllamaQuery: Send a prompt to Ollama and get a response
- OllamaChat: Multi-turn chat with Ollama models
- OllamaModelList: List available Ollama models
"""

from .ollama_node import (
    OllamaQuery,
    OllamaChat,
    OllamaModelList,
)

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

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
