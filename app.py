import os
import socket

# Force IPv4 resolution for Hugging Face and related domains to prevent WinError 10054 (IPv6 issues)
orig_getaddrinfo = socket.getaddrinfo

def patched_getaddrinfo(*args, **kwargs):
    if (
        args
        and isinstance(args[0], str)
        and (
            "huggingface" in args[0]
            or "hf.space" in args[0]
            or "hf.co" in args[0]
            or "hf-mirror.com" in args[0]
        )
    ):
        new_args = list(args)
        while len(new_args) < 3:
            new_args.append(0)
        new_args[2] = socket.AF_INET
        return orig_getaddrinfo(*new_args, **kwargs)
    return orig_getaddrinfo(*args, **kwargs)

socket.getaddrinfo = patched_getaddrinfo

from dotenv import load_dotenv

load_dotenv()

# Map LANGSMITH_* to LANGCHAIN_* to guarantee compatibility across LangChain versions
for ls_key, lc_key in [
    ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"),
    ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"),
    ("LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT"),
    ("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT"),
]:
    if ls_key in os.environ and lc_key not in os.environ:
        os.environ[lc_key] = os.environ[ls_key]

import gradio as gr
import requests
import pandas as pd
from agent import PureAgent

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

def run_and_submit_all(manual_username: str, limit_count: int, max_steps: int = 6, profile: gr.OAuthProfile | None = None):
    """
    Fetches questions, runs the PureAgent on them, maps local GAIA files,
    submits all answers, and displays the results.
    """
    # 1. Determine Username (OAuth on HF Space, or manual input for Local Run)
    if manual_username and manual_username.strip():
        username = manual_username.strip()
        print(f"User specified manual HF username: {username}")
    elif profile and getattr(profile, "username", None):
        username = f"{profile.username}"
        print(f"User logged in via HF OAuth: {username}")
    elif os.getenv("HF_USERNAME"):
        username = os.getenv("HF_USERNAME").strip()
        print(f"Using HF_USERNAME from .env: {username}")
    else:
        print("User not logged in and no username provided.")
        return "Please enter your Hugging Face username in the box above.", None


    space_id = os.getenv("SPACE_ID")
    if space_id:
        agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    else:
        agent_code = os.getenv("AGENT_CODE_URL", f"https://huggingface.co/{username}/lily-agent")

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 2. Instantiate Agent with configurable max_steps
    try:
        agent = PureAgent(max_steps=int(max_steps), verbose=True)
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None


    # 3. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            print("Fetched questions list is empty.")
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # Apply limit if specified (e.g., test 1-2 questions first)
    if limit_count and limit_count > 0:
        questions_data = questions_data[:int(limit_count)]
        print(f"Limiting evaluation to first {len(questions_data)} questions.")

    # 4. Run Agent
    results_log = []
    answers_payload = []
    print(f"\n=======================================================")
    print(f"🚀 Starting GAIA Evaluation ({len(questions_data)} Tasks)")
    print(f"=======================================================")
    
    for idx, item in enumerate(questions_data, start=1):
        task_id = item.get("task_id")
        question_text = item.get("question")
        file_name = item.get("file_name")
        level = item.get("Level", "1")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        
        print(f"\n[{idx}/{len(questions_data)}] 📋 Task ID: {task_id} (Level {level})")
        print(f"    ❓ Question: {question_text[:120]}..." if len(question_text) > 120 else f"    ❓ Question: {question_text}")
        if file_name:
            print(f"    📎 File: {file_name}")

        try:
            # PureAgent automatically resolves file_name to local GAIA/2023/ paths and logs trace steps
            submitted_answer = agent(question_text, file_name=file_name)
            trace = getattr(agent, "last_trace", {})
            tools_str = ", ".join(trace.get("tools_used", [])) if trace.get("tools_used") else "Direct"
            duration = trace.get("duration", 0.0)
            tokens = trace.get("tokens", 0)
            steps = trace.get("steps", 1)

            print(f"    🎯 Answer: '{submitted_answer}' (in {duration:.2f}s, {steps} steps, tools: [{tools_str}])")
            
            answers_payload.append(
                {"task_id": task_id, "submitted_answer": submitted_answer}
            )
            results_log.append(
                {
                    "Task ID": task_id,
                    "Question": question_text,
                    "Submitted Answer": submitted_answer,
                    "Tools Used": tools_str,
                    "Steps": steps,
                    "Duration (s)": round(duration, 2),
                    "Tokens": tokens,
                }
            )
        except Exception as e:
            print(f"    ❌ Error running agent on task {task_id}: {e}")
            results_log.append(
                {
                    "Task ID": task_id,
                    "Question": question_text,
                    "Submitted Answer": f"AGENT ERROR: {e}",
                    "Tools Used": "Error",
                    "Steps": 0,
                    "Duration (s)": 0,
                    "Tokens": 0,
                }
            )

    results_df = pd.DataFrame(results_log)
    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", results_df

    # Always save answers to a local CSV file so they are not lost!
    try:
        results_df.to_csv("submission_log.csv", index=False)
        print("\n💾 Saved detailed execution trace to submission_log.csv")
    except Exception as csv_err:
        print(f"Warning: Could not save submission_log.csv: {csv_err}")


    # 5. Prepare Submission
    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload,
    }
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 6. Submit to HF Scoring Server
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}%\n"
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        return status_message, results_df


# --- Build Gradio Interface ---
with gr.Blocks(title="Lily Agent - GAIA Unit 4 Evaluation Runner") as demo:
    gr.Markdown("# ?? Lily Agent - GAIA Evaluation & Scoring Runner")
    gr.Markdown(
        """
        Runs LilyAgent on the **Hugging Face Agents Course Unit 4** scoring benchmark.
        
        - Automatically connects to gents-course-unit4-scoring.hf.space to fetch questions.
        - Automatically maps referenced files to your local GAIA/2023/ folder.
        - Executes Lily's ReAct loop, Python tools, and Document readers.
        - Submits answers and displays your final score!
        """
    )

    with gr.Row():
        hf_username = gr.Textbox(
            label="Hugging Face Username",
            placeholder="e.g. safefunny",
            value=os.getenv("HF_USERNAME", ""),
            scale=3
        )
        question_limit = gr.Slider(
            minimum=1,
            maximum=20,
            value=2,
            step=1,
            label="Questions to Run",
            scale=2
        )
        max_steps_slider = gr.Slider(
            minimum=2,
            maximum=15,
            value=6,
            step=1,
            label="Max Steps per Task (Recommended: 5-8)",
            scale=2
        )

    run_button = gr.Button("🚀 Run Evaluation & Submit Answers", variant="primary")

    status_output = gr.Textbox(
        label="Run Status / Submission Result", lines=5, interactive=False
    )
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        inputs=[hf_username, question_limit, max_steps_slider],
        outputs=[status_output, results_table]
    )


if __name__ == "__main__":
    print("\n" + "-" * 30 + " Starting Lily Local Gradio Evaluator " + "-" * 30)
    print("Launching Gradio on http://127.0.0.1:7860 ...")
    demo.launch(debug=True, share=False)
