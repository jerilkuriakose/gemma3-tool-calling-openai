from transformers import AutoTokenizer
import json

system_prompt = """You are an expert mental health screening assistant.
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
# Test cases with different message formats
test_cases = [
    {
        "name": "Multi-turn - Depression Screening Complete Flow",
        "messages": [
            {"role": "system", "content": system_prompt},
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
    },
    {
        "name": "Multi-turn - Depression Screening Complete Flow",
        "messages": [
            {"role": "system", "content": system_prompt},
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
    },
]

# Tool definition for testing
tools = [
    {
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
                        "enum": ["PHQ-9", "GAD-7", "DAST-10", "PC-PTSD-5"],
                    },
                    "language": {
                        "type": "string",
                        "description": "Language for the questionnaire",
                        "enum": ["en", "ar"],
                    },
                },
                "required": ["test_name", "language"],
            },
        },
    }
]


def test_chat_template(tokenizer, template_path, template_name):
    """Test a chat template with various message formats"""
    print(f"\n{'=' * 60}")
    print(f"Testing: {template_name}")
    print("=" * 60)

    # Load the template
    with open(template_path, "r") as f:
        template_content = f.read()

    # Set the chat template
    tokenizer.chat_template = template_content

    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        try:
            # Test without tools
            output = tokenizer.apply_chat_template(
                test_case["messages"], tokenize=False, add_generation_prompt=True
            )
            print("WITHOUT TOOLS:")
            print(output)

            # Test with tools (only for non-image cases)
            if not any(
                isinstance(msg.get("content"), list) for msg in test_case["messages"]
            ):
                output_with_tools = tokenizer.apply_chat_template(
                    test_case["messages"],
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                print("\nWITH TOOLS:")
                print(output_with_tools)

        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {str(e)}")

        print("-" * 60)


def main():
    # Initialize tokenizer (you may need to adjust the model name)
    # Using a base model that doesn't have its own chat template
    tokenizer = AutoTokenizer.from_pretrained("/tmp/models/google/medgemma-27b-it/")

    # Test original template
    test_chat_template(
        tokenizer, "chat_template_fc_no_image.jinja", "Original Template"
    )

    # Test new template with image support
    test_chat_template(
        tokenizer, "chat_template_fc_image.jinja", "New Template with Image Support"
    )

    # Compare specific outputs side by side
    print("\n" + "=" * 60)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 60)

    for test_case in test_cases[:2]:  # Just first two cases for brevity
        print(f"\n--- {test_case['name']} ---")

        # Original template
        with open("chat_template_fc_no_image.jinja", "r") as f:
            tokenizer.chat_template = f.read()
        try:
            original_output = tokenizer.apply_chat_template(
                test_case["messages"], tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            original_output = f"ERROR: {e}"

        # New template
        with open("chat_template_fc_image.jinja", "r") as f:
            tokenizer.chat_template = f.read()
        try:
            new_output = tokenizer.apply_chat_template(
                test_case["messages"], tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            new_output = f"ERROR: {e}"

        print("\nORIGINAL:")
        print(original_output)
        print("\nNEW:")
        print(new_output)
        print("\nDIFFERENCE:", "Same" if original_output == new_output else "Different")


if __name__ == "__main__":
    main()
