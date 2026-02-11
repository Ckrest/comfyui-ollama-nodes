# Notes - ComfyUI Ollama Nodes

Brief context for agents working with this package.

## Build / Run

**Installation:**
```bash
# Clone to ComfyUI custom_nodes directory
cd ComfyUI/custom_nodes
git clone https://github.com/Ckrest/comfyui-ollama-nodes.git
systemctl --user restart comfyui  # or restart ComfyUI manually
```

**Requirements:**
- Ollama running locally on port 11434
- Models pulled via `ollama pull <model-name>`

## Path Dependencies

| Path | Purpose |
|------|---------|
| `ollama_node.py` | Main node implementation |

## Key Features

- Query any Ollama model from ComfyUI
- Full parameter control (temperature, top_p, top_k, etc.)
- System prompt support
- Model selection dropdown
