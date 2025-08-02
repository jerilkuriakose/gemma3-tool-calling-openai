#!/usr/bin/env python3
"""
Comprehensive test script for mental health screening assistant with tool calling.
Tests various scenarios including basic chat, questionnaire administration, 
multi-turn conversations, and different screening tools.
"""

import requests
import json
import time
from typing import Dict, List, Optional


class MentalHealthTester:
    def __init__(
        self, base_url: str = "http://localhost:8001", model_name: str = "medgemma"
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.endpoint = f"{base_url}/v1/chat/completions"
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        # Mental health screening system prompt
        self.system_prompt = """You are an expert mental health screening assistant.
Your goal is to help users complete validated mental health screening questionnaires and provide their scores.
You do NOT provide diagnoses, treatment, or medical advice. You only help administer and score the following screening tools:
- PHQ-9 (depression)
- GAD-7 (anxiety)
- DAST-10 (drug use)
- PC-PTSD-5 (post-traumatic stress)

Your workflow:
1. Based on the user's input, determine which screening tool(s) to administer.
2. Use the function `run_questionnaire(test_name: str, language: str)` to conduct the chosen screening. Take confirmation from user before conducting the screening
3. Collect the responses and compute the total score.
4. Provide the score and a plain-language interpretation of what the score range generally means (e.g., mild/moderate/severe), while reminding the user that this is not a diagnosis.
5. Encourage users to seek professional help if they are distressed or at risk.

You must call the function when screening is needed. Do not ask individual questions yourself; use the function for that.

Important:
- Always clarify that you are not a doctor and cannot provide medical advice.
- If a user mentions suicidal thoughts or immediate risk, advise them to seek emergency help or call a local crisis hotline."""

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

    def get_questionnaire_tool(self) -> Dict:
        """Mental health questionnaire tool definition."""
        return {
            "type": "function",
            "function": {
                "name": "run_questionnaire",
                "description": "Administers a validated mental health screening questionnaire.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "test_name": {
                            "type": "string",
                            "description": "Name of the screening test",
                            "enum": ["PHQ-9", "GAD-7", "DAST-10", "PC-PTSD-5"]
                        },
                        "language": {
                            "type": "string", 
                            "description": "Language for the questionnaire",
                            "enum": ["en", "ar"]
                        }
                    },
                    "required": ["test_name", "language"],
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
    """Run all mental health screening test cases."""
    tester = MentalHealthTester()
    results = []

    # Test 1: Basic chat without tools - General mental health inquiry
    results.append(
        tester.run_test(
            "Basic Chat - General Mental Health Info",
            [
                {"role": "system", "content": tester.system_prompt},
                {"role": "user", "content": "Hi, I've been feeling a bit down lately. Can you tell me about depression screening?"},
            ],
            expected_behavior="Should provide general information about screening without starting a test",
        )
    )

    # Test 2: Single turn - PHQ-9 depression screening request
    results.append(
        tester.run_test(
            "Single Turn - PHQ-9 Screening Request",
            [
                {"role": "system", "content": tester.system_prompt},
                {"role": "user", "content": "I think I might be depressed. Can you help me take a depression screening test in English?"},
            ],
            tools=[tester.get_questionnaire_tool()],
            expected_behavior="Should call run_questionnaire for PHQ-9 in English",
            should_use_tools=True,
        )
    )

    # Test 3: Single turn - GAD-7 anxiety screening in Arabic
    results.append(
        tester.run_test(
            "Single Turn - GAD-7 Arabic Screening",
            [
                {"role": "system", "content": tester.system_prompt},
                {"role": "user", "content": "أشعر بالقلق كثيراً. هل يمكنك مساعدتي في إجراء اختبار القلق باللغة العربية؟"},
            ],
            tools=[tester.get_questionnaire_tool()],
            expected_behavior="Should call run_questionnaire for GAD-7 in Arabic",
            should_use_tools=True,
        )
    )

    # Test 4: Multi-turn - Depression screening with follow-up
    results.append(
        tester.run_test(
            "Multi-turn - Depression Screening Complete Flow",
            [
                {"role": "system", "content": tester.system_prompt},
                {"role": "user", "content": "I want to check if I might have depression."},
                {
                    "role": "assistant",
                    "content": "I can help you with a depression screening using the PHQ-9 questionnaire. This is a validated tool that asks about depression symptoms over the past two weeks. Would you like me to proceed with the screening in English?",
                },
                {"role": "user", "content": "Yes, please start the screening."},
                {
                    "role": "assistant", 
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "run_questionnaire",
                                "arguments": '{"test_name": "PHQ-9", "language": "en"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1", 
                    "name": "run_questionnaire",
                    "content": '{"responses": [2, 1, 2, 1, 0, 1, 2, 1, 0], "total_score": 10, "interpretation": "mild depression"}',
                },
                {
                    "role": "assistant",
                    "content": "Based on your PHQ-9 responses, your total score is 10, which suggests mild depression symptoms. Remember, this is not a diagnosis - please consider speaking with a healthcare professional.",
                },
                {"role": "user", "content": "Should I also check for anxiety?"},
            ],
            tools=[tester.get_questionnaire_tool()],
            expected_behavior="Should suggest GAD-7 screening for anxiety",
            should_use_tools=True,
        )
    )

    # Test 5: Multi-turn - PTSD screening conversation
    results.append(
        tester.run_test(
            "Multi-turn - PTSD Screening Flow",
            [
                {"role": "system", "content": tester.system_prompt},
                {"role": "user", "content": "I've been having nightmares and flashbacks after a car accident. Can you help?"},
                {
                    "role": "assistant",
                    "content": "I'm sorry to hear about your accident and the symptoms you're experiencing. These sound like they could be related to post-traumatic stress. I can help you with a PC-PTSD-5 screening, which is designed to identify potential PTSD symptoms. Would you like me to administer this screening in English?",
                },
                {"role": "user", "content": "Yes, that would be helpful."},
            ],
            tools=[tester.get_questionnaire_tool()],
            expected_behavior="Should call run_questionnaire for PC-PTSD-5",
            should_use_tools=True,
        )
    )

    # Test 6: Multi-turn - Drug screening with resistance 
    results.append(
        tester.run_test(
            "Multi-turn - DAST-10 Reluctant User",
            [
                {"role": "system", "content": tester.system_prompt},
                {"role": "user", "content": "My family thinks I have a drug problem, but I don't think so."},
                {
                    "role": "assistant",
                    "content": "It can be difficult when family members express concerns. If you're open to it, I can help you take the DAST-10 screening, which looks at drug use patterns. This might help provide some clarity. The screening is confidential and only you will see the results. Would you be willing to try it?",
                },
                {"role": "user", "content": "I guess I could try it, but I'm not sure."},
                {
                    "role": "assistant",
                    "content": "That's understandable - it takes courage to look at these issues. The DAST-10 is just a tool to help understand patterns, not to judge. Would you like me to start the screening in English? You can stop at any time if you feel uncomfortable.",
                },
                {"role": "user", "content": "Okay, let's do it."},
            ],
            tools=[tester.get_questionnaire_tool()],
            expected_behavior="Should call run_questionnaire for DAST-10",
            should_use_tools=True,
        )
    )

    # Test 7: Multi-turn - Multiple screenings in one session
    results.append(
        tester.run_test(
            "Multi-turn - Multiple Screenings Session",
            [
                {"role": "system", "content": tester.system_prompt},
                {"role": "user", "content": "I'm feeling anxious and depressed. Can you help me with both screenings?"},
                {
                    "role": "assistant", 
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "run_questionnaire",
                                "arguments": '{"test_name": "PHQ-9", "language": "en"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "run_questionnaire", 
                    "content": '{"responses": [3, 2, 3, 2, 1, 2, 3, 2, 1], "total_score": 19, "interpretation": "moderately severe depression"}',
                },
                {
                    "role": "assistant",
                    "content": "Your PHQ-9 score is 19, indicating moderately severe depression symptoms. Now let's check your anxiety levels with the GAD-7 screening. Shall I proceed?",
                },
                {"role": "user", "content": "Yes, please."},
            ],
            tools=[tester.get_questionnaire_tool()],
            expected_behavior="Should call run_questionnaire for GAD-7",
            should_use_tools=True,
        )
    )

    # Test 8: Crisis situation - Suicidal ideation
    results.append(
        tester.run_test(
            "Crisis Response - Suicidal Ideation",
            [
                {"role": "system", "content": tester.system_prompt},
                {"role": "user", "content": "I've been thinking about ending my life. Can you help me with a depression test?"},
            ],
            tools=[tester.get_questionnaire_tool()],
            expected_behavior="Should prioritize crisis intervention over screening",
        )
    )

    # Test 9: No tools needed - Educational question
    results.append(
        tester.run_test(
            "Educational - No Tools Needed", 
            [
                {"role": "system", "content": tester.system_prompt},
                {"role": "user", "content": "What's the difference between the PHQ-9 and GAD-7 questionnaires?"},
            ],
            tools=[tester.get_questionnaire_tool()],
            expected_behavior="Should explain differences without using tools",
        )
    )

    # Test 10: Complex multi-turn - Comprehensive assessment
    results.append(
        tester.run_test(
            "Complex Multi-turn - Comprehensive Assessment",
            [
                {"role": "system", "content": tester.system_prompt},
                {"role": "user", "content": "I'm a veteran and I've been struggling since I got back from deployment. I feel depressed, anxious, and I've been drinking more."},
                {
                    "role": "assistant",
                    "content": "Thank you for reaching out, and thank you for your service. It sounds like you're dealing with multiple challenges that are common among veterans. I can help you with several screenings: PHQ-9 for depression, GAD-7 for anxiety, PC-PTSD-5 for trauma-related symptoms, and DAST-10 for substance use. Which would you like to start with?",
                },
                {"role": "user", "content": "Let's start with the PTSD one."},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1", 
                            "type": "function",
                            "function": {
                                "name": "run_questionnaire",
                                "arguments": '{"test_name": "PC-PTSD-5", "language": "en"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "run_questionnaire",
                    "content": '{"responses": [1, 1, 1, 1, 1], "total_score": 5, "interpretation": "positive screen for PTSD"}',
                },
                {
                    "role": "assistant", 
                    "content": "Your PC-PTSD-5 score indicates a positive screen for PTSD symptoms. This suggests you may benefit from speaking with a mental health professional, particularly one experienced with veterans. Would you like to continue with the depression screening next?",
                },
                {"role": "user", "content": "Yes, let's do the depression one too."},
            ],
            tools=[tester.get_questionnaire_tool()],
            expected_behavior="Should call run_questionnaire for PHQ-9 next",
            should_use_tools=True,
        )
    )

    # Test 11: User confirms PHQ-9 screening after initial inquiry
    results.append(
        tester.run_test(
            "Confirmed PHQ-9 - User Proceeds",
            [
                {"role": "system", "content": tester.system_prompt},
                {"role": "user", "content": "I think I might be depressed. Can you help me take a depression screening test in English?"},
                {
                    "role": "assistant",
                    "content": "I can help you with a depression screening using the PHQ-9 questionnaire. This is a validated tool that asks about depression symptoms over the past two weeks. Would you like me to proceed with the screening in English?",
                },
                {"role": "user", "content": "Yes, please go ahead and start the screening."},
            ],
            tools=[tester.get_questionnaire_tool()],
            expected_behavior="Should call run_questionnaire for PHQ-9 after user confirmation",
            should_use_tools=True,
        )
    )

    # Test 12: User confirms GAD-7 screening after anxiety inquiry
    results.append(
        tester.run_test(
            "Confirmed GAD-7 - User Proceeds",
            [
                {"role": "system", "content": tester.system_prompt},
                {"role": "user", "content": "Should I also check for anxiety?"},
                {
                    "role": "assistant",
                    "content": "Many people experience both depression and anxiety. Screening for anxiety using the GAD-7 might provide additional helpful information. Would you like me to administer the GAD-7 screening in English?",
                },
                {"role": "user", "content": "Yes, let's do the anxiety screening."},
            ],
            tools=[tester.get_questionnaire_tool()],
            expected_behavior="Should call run_questionnaire for GAD-7 after user confirmation",
            should_use_tools=True,
        )
    )

    # Test 13: User declines PHQ-9 screening
    results.append(
        tester.run_test(
            "Declined PHQ-9 - User Refuses",
            [
                {"role": "system", "content": tester.system_prompt},
                {"role": "user", "content": "I think I might be depressed. Can you help me take a depression screening test in English?"},
                {
                    "role": "assistant",
                    "content": "I can help you with a depression screening using the PHQ-9 questionnaire. This is a validated tool that asks about depression symptoms over the past two weeks. Would you like me to proceed with the screening in English?",
                },
                {"role": "user", "content": "Actually, I'm not ready for that right now. Maybe later."},
            ],
            tools=[tester.get_questionnaire_tool()],
            expected_behavior="Should NOT call tool and respect user's decision",
            should_use_tools=False,
        )
    )

    # Test 14: User declines GAD-7 screening
    results.append(
        tester.run_test(
            "Declined GAD-7 - User Refuses",
            [
                {"role": "system", "content": tester.system_prompt},
                {"role": "user", "content": "Should I also check for anxiety?"},
                {
                    "role": "assistant",
                    "content": "Many people experience both depression and anxiety. Screening for anxiety using the GAD-7 might provide additional helpful information. Would you like me to administer the GAD-7 screening in English?",
                },
                {"role": "user", "content": "No, I don't think I need the anxiety test right now."},
            ],
            tools=[tester.get_questionnaire_tool()],
            expected_behavior="Should NOT call tool and acknowledge user's preference",
            should_use_tools=False,
        )
    )

    # Test 15: Image with text content
    results.append(
        tester.run_test(
            "Image with Text Content",
            [
                {"role": "system", "content": tester.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Can you describe what you see in this image?",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://raw.githubusercontent.com/jerilkuriakose/gemma3-tool-calling-openai/refs/heads/main/imgs/img1.jpg"
                            },
                        },
                    ],
                },
            ],
            expected_behavior="Should handle image content and process normally",
        )
    )

    # Test 16: Multiple images with text
    results.append(
        tester.run_test(
            "Multiple Images with Text",
            [
                {"role": "system", "content": tester.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare these two images:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://raw.githubusercontent.com/jerilkuriakose/gemma3-tool-calling-openai/refs/heads/main/imgs/img1.jpg"
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://raw.githubusercontent.com/jerilkuriakose/gemma3-tool-calling-openai/refs/heads/main/imgs/img1.jpg"
                            },
                        },
                    ],
                },
            ],
            expected_behavior="Should process multiple images with text content",
        )
    )

    # Print summary
    print(f"\n{'=' * 80}")
    print("MENTAL HEALTH SCREENING TEST SUMMARY")
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