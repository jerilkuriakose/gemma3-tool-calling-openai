# gemma3-tool-calling-openai

# MedGemma / Gemma OpenAI-Compatible Tool Calling

This repository enables OpenAI-compatible tool calling for MedGemma / Gemma models when served with vLLM. MedGemma doesn't natively support tool calling in the OpenAI format, so this solution provides the necessary modifications to make it work seamlessly.

## Overview

MedGemma is a powerful medical language model, but it lacks built-in support for OpenAI-style function calling. This repository provides:

- **Modified Chat Template**: A custom Jinja template (`chat_template_fc.jinja`) that formats tool calls in the expected format
- **Custom Tool Parser**: A vLLM-compatible parser (`medgemma_parser.py`) that handles tool call parsing for MedGemma

## Files

- `chat_template_fc.jinja` - Modified chat template enabling OpenAI-compatible tool calling format
- `medgemma_parser.py` - Custom tool parser plugin for vLLM that works with MedGemma's output format

## Prerequisites

- vLLM installed and configured
- MedGemma model downloaded (e.g., `google/medgemma-27b-it`)
- Python environment with necessary dependencies

## Usage

### Starting vLLM Server

Use the following command to start vLLM with MedGemma and tool calling support:

```bash
vllm serve \
  /tmp/models/google/medgemma-27b-it/ \
  --port 8001 \
  --served-model-name medgemma \
  --tensor-parallel-size 2 \
  --max-model-len 16000 \
  --enable-auto-tool-choice \
  --tool-parser-plugin medgemma_parser.py \
  --tool-call-parser medgemma \
  --chat-template chat_template_fc.jinja
```

### Command Breakdown

**Tool Calling Configuration:**
```bash
--enable-auto-tool-choice \
--tool-parser-plugin medgemma_parser.py \
--tool-call-parser medgemma \
```

**Chat Template Override:**
```bash
--chat-template chat_template_fc.jinja
```

**Standard vLLM Configuration:**
```bash
--port 8001                    # API server port
--served-model-name medgemma   # Model name for API calls
--tensor-parallel-size 2       # GPU parallelization
--max-model-len 16000         # Maximum context length
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/jerilkuriakose/gemma3-tool-calling-openai.git
cd gemma3-tool-calling-openai
```

2. Ensure you have vLLM installed:
```bash
pip install vllm
```

3. Download the MedGemma model or update the path in the command to your model location.

## API Usage

Once the server is running, you can make OpenAI-compatible API calls with tool calling:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="dummy-key"  # vLLM doesn't require a real key
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_patient_info",
            "description": "Get patient information",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "The patient ID"
                    }
                },
                "required": ["patient_id"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="medgemma",
    messages=[
        {"role": "user", "content": "Get information for patient 12345"}
    ],
    tools=tools,
    tool_choice="auto"
)
```

## How It Works

1. **Chat Template Modification**: The `chat_template_fc.jinja` file modifies how MedGemma formats its responses to include proper tool call syntax that matches OpenAI's expected format.

2. **Tool Parser Integration**: The `medgemma_parser.py` provides a custom parser that vLLM uses to interpret MedGemma's tool calling outputs and convert them to the standard format.

3. **vLLM Integration**: By using the `--tool-parser-plugin` and `--tool-call-parser` flags, vLLM knows to use our custom parser for handling tool calls with this model.

## Troubleshooting

### Common Issues

- **Import Errors**: Ensure the paths to `medgemma_parser.py` and `chat_template_fc.jinja` are correct relative to where you're running vLLM
- **Model Loading**: Verify the model path exists and you have sufficient GPU memory
- **Tool Parsing**: Check vLLM logs for any parsing errors if tool calls aren't working as expected

### Debugging

Enable verbose logging to see tool parsing in action:
```bash
# Add to your vLLM command
--log-level DEBUG
```
