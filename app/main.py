"""
Claude Code - A simple coding agent implementation in Python
From CodeCrafters.io build-your-own-claude-code (Python)
"""

import argparse
import json
import os
import subprocess
import sys

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

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
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": args.p}]

    # Define tools
    tools: list[ChatCompletionToolParam] = [
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
        },
        {
            "type": "function",
            "function": {
                "name": "Write",
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path of the file to write to"
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to write to the file"
                        }
                    },
                    "required": ["file_path", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "Bash",
                "description": "Execute a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to execute"
                        }
                    },
                    "required": ["command"]
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
        assistant_message: ChatCompletionMessageParam = {"role": "assistant",
                                                         "content": message.content}
        if message.tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
                for tool_call in message.tool_calls
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
                        with open(file_path, "r", encoding="utf-8") as f:
                            file_contents = f.read()
                        tool_result = file_contents
                    except (FileNotFoundError, IOError) as e:
                        tool_result = f"Error reading file: {str(e)}"

                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })

                # Execute the Write tool
                elif function_name == "Write":
                    file_path = arguments["file_path"]
                    content = arguments["content"]
                    try:
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(content)
                        tool_result = f"Successfully wrote to {file_path}"
                    except (IOError, OSError) as e:
                        tool_result = f"Error writing file: {str(e)}"

                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })

                # Execute the Bash tool
                elif function_name == "Bash":
                    command = arguments["command"]
                    try:
                        result = subprocess.run(
                            command,
                            shell=True,
                            capture_output=True,
                            text=True,
                            timeout=30,
                            check=False
                        )
                        # Combine stdout and stderr
                        output = result.stdout
                        if result.stderr:
                            output += result.stderr
                        tool_result = output if output else "Command executed successfully"
                    except subprocess.TimeoutExpired:
                        tool_result = "Error: Command timed out after 30 seconds"
                    except (OSError, ValueError) as e:
                        tool_result = f"Error executing command: {str(e)}"

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
