import re
import json
from typing import Dict, List, Optional, Any, Union
from collections.abc import Sequence

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaMessage,
    DeltaToolCall,
    DeltaFunctionCall,
    ExtractedToolCallInformation,
    ToolCall,
    FunctionCall,
)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser,
    ToolParserManager,
)
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import random_uuid
from vllm.logger import init_logger

logger = init_logger(__name__)


@ToolParserManager.register_module(["medgemma"])
class MedGemmaToolParser(ToolParser):
    """
    Tool parser for MedGemma model that extracts tool calls from content
    in the format: ```tool_code\nprint(function_name(args))\n```
    """

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        # Pattern to match the tool call format
        self.tool_call_pattern = re.compile(r"```tool_code\s*([^`]+)```", re.DOTALL)

        # State for streaming
        self.current_tool_id = -1
        self.prev_tool_calls = []
        self.in_tool_call = False
        self.tool_buffer = ""

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from the model output for non-streaming requests.

        Expected format:
        ```tool_code
        print(function_name(param='value', param2='value2'))
        ```
        """
        logger.debug(f"Extracting tool calls from output: {model_output[:200]}...")

        # Check if tool call pattern exists
        if "```tool_code" not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            # Find all tool call matches
            matches = self.tool_call_pattern.findall(model_output)

            if not matches:
                logger.debug("No tool call matches found in pattern")
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            tool_calls = []
            for match in matches:
                tool_call = self._parse_tool_call_string(match.strip())
                if tool_call:
                    tool_calls.append(tool_call)

            # Extract content before the first tool call
            tool_start = model_output.find("```tool_code")
            content = model_output[:tool_start].strip() if tool_start > 0 else None

            logger.debug(f"Extracted {len(tool_calls)} tool calls")

            return ExtractedToolCallInformation(
                tools_called=len(tool_calls) > 0,
                tool_calls=tool_calls,
                content=content if content else None,
            )

        except Exception as e:
            logger.error(f"Error extracting tool calls: {e}")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        """
        Extract tool calls from streaming response.
        """
        logger.debug(f"Streaming delta: {repr(delta_text)}")

        # Check if we're entering a tool call
        if "```tool_code" in delta_text and not self.in_tool_call:
            self.in_tool_call = True
            self.current_tool_id += 1
            self.tool_buffer = ""
            logger.debug(f"Starting tool call {self.current_tool_id}")
            return None  # Don't stream the start of tool call

        # Check if we're in a tool call
        if self.in_tool_call:
            self.tool_buffer += delta_text

            # Check if we have a complete tool call (closing ```)
            if delta_text.count("```") > 0 and self.tool_buffer.count("```") >= 2:
                # Extract the complete tool call
                try:
                    pattern = r"```tool_code\s*([^`]+)```"
                    match = re.search(pattern, self.tool_buffer, re.DOTALL)

                    if match:
                        tool_call_str = match.group(1).strip()
                        function_name, arguments = self._parse_function_call(
                            tool_call_str
                        )

                        if function_name:
                            # Reset state
                            self.in_tool_call = False
                            self.tool_buffer = ""

                            logger.debug(
                                f"Completed tool call: {function_name} with args: {arguments}"
                            )

                            # Return the complete tool call
                            return DeltaMessage(
                                tool_calls=[
                                    DeltaToolCall(
                                        index=self.current_tool_id,
                                        type="function",
                                        id=f"chatcmpl-tool-{random_uuid()}",
                                        function=DeltaFunctionCall(
                                            name=function_name,
                                            arguments=json.dumps(
                                                arguments, ensure_ascii=False
                                            ),
                                        ),
                                    )
                                ]
                            )

                except Exception as e:
                    logger.error(f"Error parsing streaming tool call: {e}")
                    self.in_tool_call = False
                    self.tool_buffer = ""

            return None  # Don't stream tool call content

        # Regular content streaming
        return DeltaMessage(content=delta_text)

    def _parse_tool_call_string(self, tool_call_str: str) -> Optional[ToolCall]:
        """
        Parse a tool call string like: print(get_weather(location='Riyadh, Saudi Arabia'))
        """
        try:
            function_name, arguments = self._parse_function_call(tool_call_str)

            if not function_name:
                return None

            return ToolCall(
                type="function",
                function=FunctionCall(
                    name=function_name,
                    arguments=json.dumps(arguments, ensure_ascii=False),
                ),
            )

        except Exception as e:
            logger.error(f"Error parsing tool call string '{tool_call_str}': {e}")
            return None

    def _parse_function_call(
        self, tool_call_str: str
    ) -> tuple[Optional[str], Dict[str, Any]]:
        """
        Parse a function call string to extract name and arguments.
        Handles: print(get_weather(location='Riyadh, Saudi Arabia'))
        """
        tool_call_str = tool_call_str.strip()

        # Handle print() wrapper - extract the function call from inside print()
        if tool_call_str.startswith("print(") and tool_call_str.endswith(")"):
            # Extract content between print( and ) - need to be careful with nested parentheses
            inner_content = tool_call_str[6:-1]  # Remove 'print(' and last ')'
            tool_call_str = inner_content

        # Extract function name and arguments using regex
        func_pattern = r"^(\w+)\((.*)\)$"
        match = re.match(func_pattern, tool_call_str.strip(), re.DOTALL)

        if not match:
            logger.error(f"Could not parse function call: {tool_call_str}")
            return None, {}

        function_name = match.group(1)
        args_str = match.group(2).strip()

        # Parse arguments
        arguments = {}
        if args_str:
            try:
                arguments = self._parse_function_arguments(args_str)
            except Exception as e:
                logger.error(f"Error parsing arguments '{args_str}': {e}")
                return function_name, {}

        return function_name, arguments

    def _parse_function_arguments(self, args_str: str) -> Dict[str, Any]:
        """
        Parse function arguments from string format.
        Handles: location='Riyadh, Saudi Arabia', temp_unit='celsius', count=5
        """
        arguments = {}

        if not args_str.strip():
            return arguments

        # Split by commas, but be careful with quoted strings
        current_arg = ""
        in_quotes = False
        quote_char = None
        paren_level = 0

        for char in args_str + ",":  # Add comma to handle last argument
            if char in ["'", '"'] and not in_quotes:
                in_quotes = True
                quote_char = char
                current_arg += char
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
                current_arg += char
            elif char == "(" and not in_quotes:
                paren_level += 1
                current_arg += char
            elif char == ")" and not in_quotes:
                paren_level -= 1
                current_arg += char
            elif char == "," and not in_quotes and paren_level == 0:
                if current_arg.strip():
                    key, value = self._parse_single_argument(current_arg.strip())
                    if key:
                        arguments[key] = value
                current_arg = ""
            else:
                current_arg += char

        return arguments

    def _parse_single_argument(self, arg_str: str) -> tuple[Optional[str], Any]:
        """
        Parse a single argument like: location='Riyadh, Saudi Arabia'
        """
        if "=" not in arg_str:
            return None, None

        key, value = arg_str.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Remove quotes if present
        if len(value) >= 2:
            if (value.startswith("'") and value.endswith("'")) or (
                value.startswith('"') and value.endswith('"')
            ):
                value = value[1:-1]

        # Try to convert to appropriate type
        try:
            # Try boolean
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            # Try integer
            elif value.isdigit():
                value = int(value)
            # Try float
            elif "." in value and value.replace(".", "", 1).isdigit():
                value = float(value)
            # Try negative numbers
            elif value.startswith("-") and value[1:].replace(".", "", 1).isdigit():
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
        except:
            pass  # Keep as string

        return key, value


# Test function to verify the parser works with actual MedGemma output
def test_medgemma_parser():
    """Test the parser with actual MedGemma output format"""

    # Mock tokenizer for testing
    class MockTokenizer:
        def __init__(self):
            self.vocab = {}

        def __getattr__(self, name):
            return None

    try:
        parser = MedGemmaToolParser(MockTokenizer())

        # Test with actual MedGemma output
        test_output = """Here's the weather information:

```tool_code
print(get_weather(location='Riyadh, Saudi Arabia'))
```

I'll get that information for you."""

        # Mock request - create a simple dict instead of full ChatCompletionRequest
        class MockRequest:
            def __init__(self):
                self.model = "test"
                self.messages = [{"role": "user", "content": "test"}]

        request = MockRequest()

        result = parser.extract_tool_calls(test_output, request)

        print("=== Parser Test Results ===")
        print(f"Tools called: {result.tools_called}")
        print(f"Content: '{result.content}'")

        if result.tool_calls:
            for i, tool_call in enumerate(result.tool_calls):
                print(f"Tool call {i}:")
                print(f"  Function: {tool_call.function.name}")
                print(f"  Arguments: {tool_call.function.arguments}")
                # Parse and pretty print arguments
                args_dict = json.loads(tool_call.function.arguments)
                print(f"  Parsed args: {args_dict}")
        else:
            print("No tool calls found!")

        return result

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    test_medgemma_parser()
