import sys
import os
import time

# Reconfigure stdout/stderr encoding for Windows console emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure the root project directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.agent import LilyAgent
from src.utils import clean_markdown_text

class GAIABenchmarkAgent:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        # Unique session for each evaluator run
        self.agent = LilyAgent(session_id="gaia_benchmark_run", max_steps=15)
        self.last_trace = {
            "tools_used": [],
            "steps": 0,
            "duration": 0.0,
            "tokens": 0,
            "in_tokens": 0,
            "out_tokens": 0,
            "speed": 0.0,
        }
        
    def __call__(self, question: str, file_name: str | None = None, file_path: str | None = None) -> str:
        start_time = time.perf_counter()
        self.last_trace = {
            "tools_used": [],
            "steps": 0,
            "duration": 0.0,
            "tokens": 0,
            "in_tokens": 0,
            "out_tokens": 0,
            "speed": 0.0,
        }

        prompt = ""
        resolved_file = None
        
        if file_path and os.path.exists(file_path):
            resolved_file = file_path
        elif file_name:
            # Check local GAIA directories
            candidates = [
                os.path.abspath(file_name),
                os.path.abspath(os.path.join("GAIA", "2023", "validation", file_name)),
                os.path.abspath(os.path.join("GAIA", "2023", "test", file_name)),
                os.path.abspath(os.path.join("GAIA", file_name)),
            ]
            for c in candidates:
                if os.path.exists(c):
                    resolved_file = c
                    break
        
        if resolved_file:
            ext = resolved_file.lower().split('.')[-1]
            if ext in ['png', 'jpg', 'jpeg', 'webp', 'bmp', 'gif']:
                prompt += f"Attached Image Context: You have access to a local image at '{resolved_file}'. Use the `inspect_image` tool with specific questions to view, read, and analyze it.\n\n"
            else:
                prompt += f"Attached File Context: You have access to a local file at '{resolved_file}'. Use the `read_document` tool to inspect it.\n\n"
        elif file_name:
            prompt += f"Attached File Context: The question references file '{file_name}', but it could not be located locally.\n\n"
            
        prompt += f"Question: {question}\n\n"
        
        prompt += (
            "CRITICAL INSTRUCTION: You are taking a strict exact-match benchmark test (GAIA). "
            "You MUST output ONLY the final answer without ANY conversational filler, pleasantries, or explanations. "
            "For example, if the answer is 42, just output '42'. If the answer is 'John Doe', output 'John Doe'. "
            "Do not write 'The answer is 42' or 'Here is the result'.\n"
            "If you need to view or read an image, use the `inspect_image` tool. "
            "If you need to calculate something or manipulate data, use the `execute_python` tool. "
            "If you need to look up facts, use `search_web` or `fetch_web_page`. "
            "Take a deep breath and think step-by-step."
        )
        
        # Reset intermediate conversation turns to keep tasks isolated
        self.agent.reset_context()

        raw_answer = ""
        tools_called = []
        max_step_reached = 0

        # Execute streaming with live trace logging
        for event_type, data in self.agent.chat_stream(prompt):
            if event_type == "token":
                raw_answer += data
                
            elif event_type == "tool_start":
                # Clear answer buffer as this text was pre-tool thinking
                raw_answer = ""
                func_name = data.get("name", "")
                func_args = data.get("args", {})
                step_num = data.get("step", 1)
                max_step_reached = max(max_step_reached, step_num)
                
                if func_name not in tools_called:
                    tools_called.append(func_name)
                    
                if self.verbose:
                    print(f"    🔍 [Step {step_num}] Calling Tool: {func_name}")
                    if func_name == "execute_python" and "code" in func_args:
                        code_snippet = func_args['code'].strip().replace('\n', '\n        ')
                        print(f"        └─ Code:\n        {code_snippet}")
                    else:
                        print(f"        └─ Args: {func_args}")
                        
            elif event_type == "tool_end":
                func_name = data.get("name", "")
                duration = data.get("duration", 0.0)
                char_count = data.get("char_count", 0)
                if self.verbose:
                    print(f"    ✔ Finished {func_name} in {duration:.2f}s ({char_count:,} chars)")
                    
            elif event_type == "usage":
                total_tok = data.get("total_tokens", 0)
                in_tok = data.get("input_tokens", 0)
                out_tok = data.get("output_tokens", 0)
                speed = data.get("tokens_per_sec", 0.0)
                dur = data.get("duration_sec", 0.0)
                self.last_trace.update({
                    "tokens": total_tok,
                    "in_tokens": in_tok,
                    "out_tokens": out_tok,
                    "speed": speed,
                })
                if self.verbose:
                    print(f"    ⚡ {total_tok:,} tokens ({in_tok:,} in / {out_tok:,} out) • {speed:.1f} tok/s")
                    
            elif event_type == "error":
                if self.verbose:
                    print(f"    ❌ Error during step: {data}")

        total_duration = time.perf_counter() - start_time
        clean_ans = clean_markdown_text(raw_answer.strip())

        self.last_trace.update({
            "tools_used": tools_called,
            "steps": max_step_reached if max_step_reached > 0 else 1,
            "duration": total_duration,
        })

        return clean_ans


