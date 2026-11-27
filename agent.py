"""Vision agent driven by Harbor from the evaluation host."""

import asyncio
import json
import os
from pathlib import Path

from harbor.agents.base import BaseAgent

from cl_adapter import Vision, load_key, parse_continual_learning

SHELL_TOOL = {
    "type": "function",
    "function": {
        "name": "run_command",
        "description": "Execute a shell command in the current benchmark sandbox. Files persist across calls.",
        "parameters": {
            "type": "object", "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}


class VisionAgent(BaseAgent):
    @staticmethod
    def name():
        return "vision-cl"

    def version(self):
        return "1"

    async def setup(self, environment):
        pass

    async def run(self, instruction, environment, context):
        continual_learning = parse_continual_learning(os.environ.get("VISION_CONTINUAL_LEARNING", "on"))
        vision = Vision(load_key(Path(__file__).resolve().parent), continual_learning=continual_learning)
        messages = [
            {"role": "system", "content": (
                "Complete the requested task in the provided sandbox through run_command. "
                "Keep all work within the task environment and leave the requested artifacts there. "
                "Do not inspect or modify the evaluation infrastructure. "
                "When finished, respond with a concise summary."
            )},
            {"role": "user", "content": instruction},
        ]
        trace_path = self.logs_dir / "vision-trajectory.jsonl"
        try:
            for _ in range(500):
                message, _ = await asyncio.to_thread(vision.chat, messages, [SHELL_TOOL])
                messages.append(message)
                calls = message.get("tool_calls", [])
                with trace_path.open("a", encoding="utf-8") as trace:
                    trace.write(json.dumps(message) + "\n")
                    for call in calls:
                        arguments = json.loads(call["function"]["arguments"])
                        if call["function"]["name"] != "run_command" or not isinstance(arguments.get("command"), str):
                            output = "Invalid run_command call"
                        else:
                            result = await environment.exec(arguments["command"], timeout_sec=120)
                            output = json.dumps({
                                "exit_code": result.return_code,
                                "stdout": (result.stdout or "")[-30000:],
                                "stderr": (result.stderr or "")[-10000:],
                            })
                        tool_message = {"role": "tool", "tool_call_id": call["id"], "content": output}
                        messages.append(tool_message)
                        trace.write(json.dumps(tool_message) + "\n")
                if not calls:
                    break
        finally:
            report = vision.report()
            context.n_input_tokens = report["input_tokens"]
            context.n_output_tokens = report["output_tokens"]
            (self.logs_dir / "vision-usage.json").write_text(json.dumps(report, indent=2) + "\n")
