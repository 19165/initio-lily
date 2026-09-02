import sys
import os

# Ensure the root project directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import LilyAgent

class GAIABenchmarkAgent:
    def __init__(self):
        self.agent = LilyAgent(session_id="gaia_benchmark_run", max_steps=15)
        
    def __call__(self, question: str, file_name: str | None = None, file_path: str | None = None) -> str:
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
            prompt += f"Attached File Context: You have access to a local file at '{resolved_file}'. Use the 'read_document' tool to inspect it.\n\n"
        elif file_name:
            prompt += f"Attached File Context: The question references file '{file_name}', but it could not be located locally.\n\n"
            
        prompt += f"Question: {question}\n\n"
        
        prompt += (
            "CRITICAL INSTRUCTION: You are taking a strict exact-match benchmark test (GAIA). "
            "You MUST output ONLY the final answer without ANY conversational filler, pleasantries, or explanations. "
            "For example, if the answer is 42, just output '42'. If the answer is 'John Doe', output 'John Doe'. "
            "Do not write 'The answer is 42' or 'Here is the result'.\n"
            "If you need to calculate something or manipulate data, use the execute_python tool. "
            "If you need to look up facts, use search_web or fetch_web_page. "
            "Take a deep breath and think step-by-step."
        )
        raw_answer = self.agent.chat(prompt)
        return raw_answer.strip()

