"""
Claude Code - A simple coding agent implementation in Python
From CodeCrafters.io build-your-own-claude-code (Python)
"""

import argparse
import json
import os
import sys

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")


def main():
    """Main function to run the Claude Code agent."""
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # Initialize conversation with user's prompt
    messages = [{"role": "user", "content": args.p}]

    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Read and return the contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path to the file to read"
                        }
                    },
                    "required": ["file_path"]
                }
            }
        }
    ]

    # Agent loop
    while True:
        # Send current conversation to the model
        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
            tools=tools,
        )

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")

        choice = chat.choices[0]
        message = choice.message

        # Build assistant message for conversation history
        assistant_message = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]

        messages.append(assistant_message)

        # Check if there are tool calls
        if message.tool_calls and len(message.tool_calls) > 0:
            # Execute each tool call
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                arguments_json = tool_call.function.arguments
                arguments = json.loads(arguments_json)

                # Execute the Read tool
                if function_name == "Read":
                    file_path = arguments["file_path"]
                    try:
                        with open(file_path, "r") as f:
                            file_contents = f.read()
                        tool_result = file_contents
                    except Exception as e:
                        tool_result = f"Error reading file: {str(e)}"

                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })
            # Continue loop to send tool results back
        else:
            # No tool calls, we have the final response
            if message.content:
                print(message.content, end="")
            break


if __name__ == "__main__":
    main()
