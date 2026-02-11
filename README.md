# ComfyUI Ollama Nodes

![AI-Assisted](https://img.shields.io/badge/AI-Assisted-blue)

Local LLM integration for ComfyUI via [Ollama](https://ollama.ai).

## Features

- Query local Ollama models from within ComfyUI workflows
- Full parameter control (temperature, top_k, top_p, etc.)
- Multi-turn chat support
- Automatic model list detection

## Nodes

| Node | Purpose |
|------|---------|
| **Ollama Query** | Send a prompt, get a response |
| **Ollama Chat** | Multi-turn conversation with message history |
| **Ollama Model List** | List available models on your Ollama instance |

## Requirements

- [Ollama](https://ollama.ai) installed and running
- At least one model pulled (`ollama pull llama3.2`)
- Python `requests` library: `pip install requests>=2.28`

## Installation

1. Clone or download to `ComfyUI/custom_nodes/`:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/Ckrest/comfyui-ollama-nodes.git
   ```

2. Restart ComfyUI

3. Nodes appear under **Ollama** category

## Usage

### Basic Query

1. Add **Ollama Query** node
2. Select model from dropdown (auto-populated from Ollama)
3. Enter prompt
4. Connect output to your workflow

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| temperature | 0.7 | Higher = more creative |
| top_k | 40 | Sample from top K tokens |
| top_p | 0.9 | Nucleus sampling threshold |
| max_tokens | 256 | Maximum response length |
| seed | -1 | Set >= 0 for reproducible output |

### Chat Mode

Use **Ollama Chat** for multi-turn conversations:
- `user_message` - Your current message
- `messages_json` - Previous conversation (connect from output)
- Returns updated message history for chaining

## Configuration

By default, connects to `http://127.0.0.1:11434`. Override with the `ollama_host` parameter if Ollama runs elsewhere.

## License

MIT License - See [LICENSE](LICENSE)
