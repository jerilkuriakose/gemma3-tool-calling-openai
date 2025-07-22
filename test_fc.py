#!/usr/bin/env python3
"""
Comprehensive test script for model deployment with and without function calling.
Tests various scenarios including basic chat, tool usage, multi-turn conversations, and edge cases.
"""

import requests
import json
import time
from typing import Dict, List, Any, Optional


class ModelTester:
    def __init__(
        self, base_url: str = "http://localhost:8001", model_name: str = "medgemma"
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.endpoint = f"{base_url}/v1/chat/completions"
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def make_request(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Dict:
        """Make a request to the model API."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        try:
            response = self.session.post(self.endpoint, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def get_weather_tool(self) -> Dict:
        """Standard weather tool definition."""
        return {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a given city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country, e.g., Riyadh, Saudi Arabia",
                        }
                    },
                    "required": ["location"],
                },
            },
        }

    def get_calculator_tool(self) -> Dict:
        """Calculator tool definition."""
        return {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform basic mathematical calculations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate, e.g., '2 + 3 * 4'",
                        }
                    },
                    "required": ["expression"],
                },
            },
        }

    def search_tool(self) -> Dict:
        """Search tool definition."""
        return {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def run_test(
        self,
        test_name: str,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        expected_behavior: str = "",
        should_use_tools: bool = False,
    ) -> Dict:
        """Run a single test case."""
        print(f"\n{'=' * 60}")
        print(f"TEST: {test_name}")
        print(f"Expected: {expected_behavior}")
        print(f"{'=' * 60}")

        # Print request details
        print(f"Messages: {json.dumps(messages, indent=2)}")
        if tools:
            print(f"Tools available: {[tool['function']['name'] for tool in tools]}")

        # Make request
        start_time = time.time()
        response = self.make_request(messages, tools)
        end_time = time.time()

        # Print response
        print(f"\nResponse time: {end_time - start_time:.2f}s")
        print(f"Response: {json.dumps(response, indent=2)}")

        # Analyze response
        analysis = self.analyze_response(response, should_use_tools)
        print(f"\nAnalysis: {analysis}")

        return {
            "test_name": test_name,
            "response": response,
            "analysis": analysis,
            "response_time": end_time - start_time,
        }

    def analyze_response(self, response: Dict, should_use_tools: bool = False) -> str:
        """Analyze the response and provide feedback."""
        if "error" in response:
            return f"❌ Request failed: {response['error']}"

        if "choices" not in response or not response["choices"]:
            return "❌ No choices in response"

        choice = response["choices"][0]
        message = choice.get("message", {})

        has_content = bool(message.get("content"))
        has_tool_calls = bool(message.get("tool_calls"))

        if should_use_tools and not has_tool_calls:
            return "⚠️  Expected tool call but none made"
        elif not should_use_tools and has_tool_calls:
            return "⚠️  Unexpected tool call made"
        elif has_tool_calls:
            tool_names = [tc["function"]["name"] for tc in message["tool_calls"]]
            return f"✅ Tool calls made: {tool_names}"
        elif has_content:
            return f"✅ Response generated: {len(message['content'])} chars"
        else:
            return "❌ No content or tool calls in response"


def main():
    """Run all test cases."""
    tester = ModelTester()
    results = []

    # Test 1: Basic chat without tools
    results.append(
        tester.run_test(
            "Basic Chat - No Tools",
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! How are you today?"},
            ],
            expected_behavior="Should respond normally without tool calls",
        )
    )

    # Test 2: Simple tool call - Weather
    results.append(
        tester.run_test(
            "Weather Tool Call",
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the weather in Riyadh today?"},
            ],
            tools=[tester.get_weather_tool()],
            expected_behavior="Should call get_weather with Riyadh location",
            should_use_tools=True,
        )
    )

    # Test 3: Multi-turn conversation with tool
    results.append(
        tester.run_test(
            "Multi-turn Weather Conversation",
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the weather in Riyadh today?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "Riyadh, Saudi Arabia"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "get_weather",
                    "content": "42°C with clear skies and light breeze",
                },
                {
                    "role": "assistant",
                    "content": "The weather in Riyadh today is 42°C with clear skies and light breeze.",
                },
                {"role": "user", "content": "What about Dammam?"},
            ],
            tools=[tester.get_weather_tool()],
            expected_behavior="Should call get_weather for Dammam",
            should_use_tools=True,
        )
    )

    # Test 4: Calculator tool
    results.append(
        tester.run_test(
            "Calculator Tool",
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant with access to a calculator.",
                },
                {"role": "user", "content": "What is 15 * 23 + 45?"},
            ],
            tools=[tester.get_calculator_tool()],
            expected_behavior="Should use calculator tool for math",
            should_use_tools=True,
        )
    )

    # Test 5: Multiple tools available
    results.append(
        tester.run_test(
            "Multiple Tools - Weather Request",
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's the weather like in Tokyo?"},
            ],
            tools=[
                tester.get_weather_tool(),
                tester.get_calculator_tool(),
                tester.search_tool(),
            ],
            expected_behavior="Should choose weather tool from multiple options",
            should_use_tools=True,
        )
    )

    # Test 6: Ambiguous request with tools
    results.append(
        tester.run_test(
            "Ambiguous Request",
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Tell me something interesting about the moon.",
                },
            ],
            tools=[tester.search_tool()],
            expected_behavior="May or may not use search tool",
        )
    )

    # Test 7: No tool needed despite availability
    results.append(
        tester.run_test(
            "Tools Available But Not Needed",
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain what photosynthesis is."},
            ],
            tools=[tester.get_weather_tool(), tester.search_tool()],
            expected_behavior="Should answer directly without tools",
        )
    )

    # Test 8: Complex tool arguments
    results.append(
        tester.run_test(
            "Complex Tool Arguments",
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Search for recent news about artificial intelligence, but only show me 3 results.",
                },
            ],
            tools=[tester.search_tool()],
            expected_behavior="Should call search with query and max_results=3",
            should_use_tools=True,
        )
    )

    # Test 9: Edge case - Empty user message
    results.append(
        tester.run_test(
            "Edge Case - Empty Message",
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": ""},
            ],
            expected_behavior="Should handle gracefully",
        )
    )

    # Test 10: Long conversation context
    results.append(
        tester.run_test(
            "Long Context Conversation",
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "I'm planning a trip to Saudi Arabia."},
                {
                    "role": "assistant",
                    "content": "That sounds exciting! Saudi Arabia has many beautiful places to visit.",
                },
                {"role": "user", "content": "What cities should I visit?"},
                {
                    "role": "assistant",
                    "content": "I'd recommend Riyadh, Jeddah, and Dammam for starters.",
                },
                {
                    "role": "user",
                    "content": "What's the weather like in Riyadh right now?",
                },
            ],
            tools=[tester.get_weather_tool()],
            expected_behavior="Should use context and call weather tool",
            should_use_tools=True,
        )
    )

    # Print summary
    print(f"\n{'=' * 80}")
    print("TEST SUMMARY")
    print(f"{'=' * 80}")

    total_tests = len(results)
    successful_tests = sum(1 for r in results if "✅" in r["analysis"])
    failed_tests = sum(1 for r in results if "❌" in r["analysis"])
    warning_tests = sum(1 for r in results if "⚠️" in r["analysis"])

    print(f"Total tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Warnings: {warning_tests}")
    print(
        f"Average response time: {sum(r['response_time'] for r in results) / total_tests:.2f}s"
    )

    # Print detailed results
    for result in results:
        status = (
            "✅"
            if "✅" in result["analysis"]
            else "❌"
            if "❌" in result["analysis"]
            else "⚠️"
        )
        print(f"{status} {result['test_name']}: {result['response_time']:.2f}s")


if __name__ == "__main__":
    main()
