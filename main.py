"""Interactive Terminal CLI for Lily AI Assistant.

This module provides a rich, minimalist command-line interface featuring:
- Live streaming markdown rendering via Rich Live.
- Interactive prompt with persistent history (.lily_history) and auto-completion.
- Built-in slash commands (/tools, /facts, /tasks, /model, /stats, /debug, /help, /clear, /exit).
- Token usage tracing & metrics (real-time tok/s, prompt vs completion stats).
- Debug & Tracing Mode (--debug or /debug) for card-based tool telemetry and reasoning steps.
- Windows file drag-and-drop normalization.
- Live PostgreSQL database & Multimodal Vision status indicator.
"""

import sys
import os

# -------------------------------------------------------------
# 0. Windows Console Encoding Fix
# -------------------------------------------------------------
# ป้องกันปัญหา UnicodeEncodeError บน Windows เมื่อพิมพ์ข้อความภาษาไทยหรือ Emoji
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")

# -------------------------------------------------------------
# 1. External Libraries Import
# -------------------------------------------------------------
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from rich.live import Live
from rich.syntax import Syntax
from rich import box
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML

from src.agent import LilyAgent
from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_VISION_MODEL
from src.tools import ALL_TOOLS, complete_task, get_facts, get_tasks, list_notes, read_note
from src.utils import clean_markdown_text

# Global Rich Console Instance
console = Console()

# -------------------------------------------------------------
# 2. UI Styling & Constants
# -------------------------------------------------------------
LILY_ASCII_LOGO = r"""
   __ _ __     
  / /(_) /_ __ 
 / // / / // / 
/_//_/_/\_, /  
       /___/   
"""

# คำสั่งลัด (Slash Commands) พร้อมคำอธิบาย
SLASH_COMMANDS = {
    "/tools": "List all 17 available agent tools & capabilities",
    "/notes": "View all saved local markdown notes in knowledge base",
    "/facts": "View all remembered facts about you from memory",
    "/memory": "Alias for /facts",
    "/tasks": "List all active tasks from Todoist",
    "/today": "Show tasks due today or overdue",
    "/done": "Complete a task by name or ID (/done <title/id>)",
    "/model": "Show active LLM, Vision Model, and system info",
    "/stats": "Show token usage & performance analytics",
    "/debug": "Toggle live debug / tool telemetry mode",
    "/clear": "Clear screen",
    "/help": "Show commands and shortcut cheatsheet",
    "/exit": "Exit assistant",
}

slash_completer = WordCompleter(
    words=list(SLASH_COMMANDS.keys()),
    meta_dict=SLASH_COMMANDS,
    ignore_case=True,
    match_middle=True,
)

prompt_style = Style.from_dict(
    {
        "prompt": "#88c0d0 bold",
        "completion-menu.completion": "bg:#2e3440 #d8dee9",
        "completion-menu.completion.current": "bg:#81a1c1 #2e3440 bold",
        "completion-menu.meta.completion": "bg:#3b4252 #949ca8",
        "completion-menu.meta.completion.current": "bg:#5e81ac #eceff4",
    }
)


# -------------------------------------------------------------
# 3. Helper Renderers
# -------------------------------------------------------------
def print_banner(agent: LilyAgent, debug_mode: bool = False):
    """แสดงโลโก้หัวโปรแกรม สถานะ Database โมเดลที่ใช้งาน และโหมด Debug."""
    logo = Text(LILY_ASCII_LOGO.strip("\n"), style="bold magenta")
    console.print(logo)

    db_badge = (
        "[bold green]● db online[/bold green]"
        if agent.db_available
        else "[bold yellow]○ db offline[/bold yellow]"
    )
    tools_count = len(ALL_TOOLS)
    debug_badge = " • [bold cyan][DEBUG MODE][/bold cyan]" if debug_mode else ""

    console.print(
        f"[dim]lily v0.2 • model: [white]{OLLAMA_MODEL}[/white] • {db_badge} • [cyan]{tools_count} tools ready[/cyan]{debug_badge} • type [cyan]/[/cyan] for commands[/dim]\n"
    )


def render_tools_table():
    """แสดงรายการ Capabilities และ Tool Catalog ทั้งหมด 17 ตัวของ Lily."""
    table = Table(
        box=box.ROUNDED,
        border_style="dim magenta",
        show_header=True,
        header_style="bold magenta",
        title="[bold magenta]🛠️ Lily Agent Tool Catalog (17 Capabilities)[/bold magenta]",
    )
    table.add_column("Category", style="bold cyan", width=18)
    table.add_column("Tool Name", style="white", width=22)
    table.add_column("Description", style="dim")

    categories = [
        ("👁️ Vision", "inspect_image", "Analyze local images (.png, .jpg) with Multimodal Vision"),
        ("📄 Documents", "read_document", "Extract text from PDF, CSV, Excel, TXT, JSON, MD files"),
        ("🐍 Python REPL", "execute_python", "Execute Python code in isolated sandbox (15s timeout)"),
        ("🌐 Web & Search", "search_web", "Real-time web search via Tavily Search API"),
        ("🌐 Web & Search", "fetch_web_page", "Deep clean markdown webpage extraction via Tavily Extract"),
        ("📝 Local Notes", "create_or_update_note", "Save/overwrite Markdown note in local notes/ dir"),
        ("📝 Local Notes", "append_to_note", "Append timestamped content/logs to existing note"),
        ("📝 Local Notes", "read_note", "Read content of a local Markdown note"),
        ("📝 Local Notes", "list_notes", "List all Markdown files in local notes directory"),
        ("📝 Local Notes", "search_notes", "Full-text keyword search across all local notes"),
        ("📋 Tasks (Todoist)", "add_task", "Create new task with natural language due date"),
        ("📋 Tasks (Todoist)", "get_tasks", "Query active tasks using natural filters (today, overdue)"),
        ("📋 Tasks (Todoist)", "complete_task", "Mark task as completed by title or unique ID"),
        ("💾 Memory (DB)", "save_fact", "Persist user facts/preferences into PostgreSQL memory"),
        ("💾 Memory (DB)", "get_facts", "Retrieve remembered facts by category"),
        ("💾 Memory (DB)", "delete_fact_by_key", "Remove specific fact from persistent memory"),
        ("⏰ System / Time", "get_current_time", "Get live system date, timestamp, and timezone"),
    ]

    for cat, name, desc in categories:
        table.add_row(cat, name, desc)

    console.print(table)
    console.print()


def render_notes_table():
    """ดึงข้อมูลรายการไฟล์โน้ต Markdown ในโฟลเดอร์ notes/ และแสดงผลเป็น Table."""
    with console.status("[dim]scanning local notes...[/dim]", spinner="dots"):
        raw_notes = list_notes.invoke({})

    if "No local notes found" in raw_notes:
        console.print("[dim yellow]No local notes in 'notes/' directory yet. Ask Lily to create one![/dim yellow]\n")
        return

    table = Table(
        box=box.ROUNDED,
        border_style="dim blue",
        show_header=True,
        header_style="bold blue",
        title="[bold blue]📝 Local Notes Knowledge Base[/bold blue]",
    )
    table.add_column("Filename", style="bold white", width=28)
    table.add_column("Size", style="cyan", width=12)
    table.add_column("Last Modified", style="dim")

    for line in raw_notes.split("\n"):
        if line.startswith("- 📄 **"):
            try:
                # Format: - 📄 **filename.md** (0.5 KB, Modified: 2026-09-02 22:40)
                fname_part, meta_part = line[7:].split("** (", 1)
                meta_part = meta_part.rstrip(")")
                size_str, mtime_str = meta_part.split(", Modified: ")
                table.add_row(fname_part.strip(), size_str.strip(), mtime_str.strip())
            except Exception:
                continue

    console.print(table)
    console.print()


def render_facts_table():
    """ดึงข้อมูลความจำส่วนตัว (User Facts) จาก PostgreSQL และแสดงผลเป็น Table."""
    with console.status("[dim]fetching memory facts...[/dim]", spinner="dots"):
        raw_facts = get_facts.invoke({})

    if "No saved facts found" in raw_facts or not raw_facts.strip():
        console.print("[dim yellow]No saved facts in memory yet. You can tell Lily to remember things![/dim yellow]\n")
        return

    table = Table(
        box=box.ROUNDED,
        border_style="dim green",
        show_header=True,
        header_style="bold green",
        title="[bold green]💾 Remembered User Facts & Preferences[/bold green]",
    )
    table.add_column("Category", style="cyan", width=14)
    table.add_column("Key", style="bold white", width=22)
    table.add_column("Value", style="white")

    for line in raw_facts.split("\n"):
        if line.startswith("- ["):
            try:
                # Format: - [category] key: value
                cat_part, rest = line[3:].split("] ", 1)
                key, val = rest.split(": ", 1)
                table.add_row(cat_part.strip(), key.strip(), val.strip())
            except Exception:
                continue

    console.print(table)
    console.print()


def render_model_info(agent: LilyAgent):
    """แสดงข้อมูลสถาปัตยกรรม โมเดล และสถานะระบบ."""
    table = Table(box=None, show_header=False, padding=(0, 2))
    table.add_column("Property", style="dim white")
    table.add_column("Value", style="bold cyan")

    db_status = "Connected (Online)" if agent.db_available else "Disconnected (Offline)"
    table.add_row("Main LLM", OLLAMA_MODEL)
    table.add_row("Vision Engine", f"{OLLAMA_VISION_MODEL} (Multimodal)")
    table.add_row("Ollama Host", OLLAMA_BASE_URL)
    table.add_row("PostgreSQL DB", db_status)
    table.add_row("Active Session ID", agent.session_id)
    table.add_row("Memory Window", f"{agent.window_size} messages")
    table.add_row("Max ReAct Steps", f"{agent.max_steps} turns per request")

    panel = Panel(
        table,
        title="[bold cyan]🤖 System & Model Architecture[/bold cyan]",
        border_style="dim cyan",
        box=box.ROUNDED,
        padding=(1, 2),
    )
    console.print(panel)
    console.print()


def render_help_menu():
    """แสดงคู่มือการใช้งานและ Cheatsheet."""
    table = Table(
        box=box.ROUNDED,
        border_style="dim yellow",
        show_header=True,
        header_style="bold yellow",
        title="[bold yellow]📖 Lily CLI Command Cheatsheet[/bold yellow]",
    )
    table.add_column("Command", style="bold cyan", width=16)
    table.add_column("Description", style="white")

    for cmd, desc in SLASH_COMMANDS.items():
        table.add_row(cmd, desc)

    console.print(table)
    console.print("[dim]💡 Tip: You can drag and drop any file (.png, .pdf, .csv, .xlsx) into the terminal to ask questions about it![/dim]\n")


def render_tasks_table(filter_query: str = ""):
    """ดึงข้อมูลจาก Todoist และแสดงผลเป็น Table บน Terminal."""
    with console.status("[dim]fetching tasks...[/dim]", spinner="dots"):
        raw_result = get_tasks.invoke({"filter_query": filter_query})

    if "No tasks found" in raw_result or not raw_result.strip():
        console.print("[dim yellow]No active tasks found.[/dim yellow]\n")
        return

    table = Table(
        box=box.ROUNDED,
        border_style="dim white",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("ID", style="dim", width=18)
    table.add_column("Task", style="white")
    table.add_column("Due", style="yellow")

    for line in raw_result.split("\n"):
        if line.startswith("- [ID:"):
            parts = line.split("] ", 1)
            task_id = parts[0].replace("- [ID:", "").strip()
            rest = parts[1] if len(parts) > 1 else ""

            if "(Due:" in rest:
                content, due = rest.rsplit("(Due:", 1)
                due = due.rstrip(")")
            else:
                content, due = rest, "-"

            table.add_row(task_id, content.strip(), due.strip())

    console.print(table)
    console.print()


def render_session_stats(agent: LilyAgent):
    """แสดงสถิติการใช้ Token และประสิทธิภาพความเร็วของ Session ปัจจุบัน."""
    stats = agent.get_session_stats()
    total_tok = stats.get("total_tokens", 0)
    in_tok = stats.get("total_input_tokens", 0)
    out_tok = stats.get("total_output_tokens", 0)
    turns = stats.get("total_turns", 0)
    speed = stats.get("avg_tokens_per_sec", 0.0)

    in_pct = (in_tok / total_tok * 100) if total_tok > 0 else 0
    out_pct = (out_tok / total_tok * 100) if total_tok > 0 else 0

    table = Table(box=None, show_header=False, padding=(0, 2))
    table.add_column("Metric", style="dim white")
    table.add_column("Value", style="bold cyan")

    table.add_row("Active Session", str(stats.get("active_session_id", "default")))
    table.add_row("Context Messages", str(stats.get("working_context_messages", 0)))
    table.add_row("Conversation Turns", f"{turns:,}")
    table.add_row("Total Tokens Used", f"{total_tok:,}")
    table.add_row("  ├─ Prompt (In)", f"{in_tok:,} ({in_pct:.1f}%)")
    table.add_row("  └─ Generated (Out)", f"{out_tok:,} ({out_pct:.1f}%)")
    table.add_row("Avg Generation Speed", f"{speed:.1f} tok/s")

    panel = Panel(
        table,
        title="[bold magenta]📊 Session Token Analytics[/bold magenta]",
        border_style="dim magenta",
        box=box.ROUNDED,
        padding=(1, 2),
    )
    console.print(panel)
    console.print()


# -------------------------------------------------------------
# 4. Main Application Loop
# -------------------------------------------------------------
def main():
    """Entry point หลักสำหรับรัน Interactive Chat CLI."""
    # ตรวจสอบ Flag --debug จากคำสั่งตอนเริ่มรัน (เช่น python main.py --debug)
    debug_mode = "--debug" in sys.argv or "-d" in sys.argv

    # สร้าง Instance ของ Agent
    agent = LilyAgent()

    # สร้าง Prompt Session พร้อม Persistent History ในไฟล์ .lily_history
    session = PromptSession(
        completer=slash_completer,
        style=prompt_style,
        history=FileHistory(".lily_history"),
    )

    console.clear()
    print_banner(agent, debug_mode)

    while True:
        try:
            user_input = session.prompt(HTML("<ansicyan>❯ </ansicyan>")).strip()
            if not user_input:
                continue

            # ==================================================
            # 1. จัดการ Slash Commands
            # ==================================================
            cmd = user_input.lower()

            if cmd in ["/exit", "/quit", ":q"]:
                console.print("[dim magenta]bye! 🌸[/dim magenta]")
                break

            elif cmd == "/clear":
                console.clear()
                print_banner(agent, debug_mode)
                continue

            elif cmd == "/debug":
                debug_mode = not debug_mode
                status_str = (
                    "[bold cyan]ON[/bold cyan]"
                    if debug_mode
                    else "[dim]OFF[/dim]"
                )
                console.print(f"[dim]⚡ Debug / Tracing Mode: {status_str}[/dim]\n")
                continue

            elif cmd in ["/tools", "/capabilities"]:
                render_tools_table()
                continue

            elif cmd in ["/notes", "/note"]:
                render_notes_table()
                continue

            elif cmd in ["/facts", "/memory"]:
                render_facts_table()
                continue

            elif cmd in ["/model", "/info", "/system"]:
                render_model_info(agent)
                continue

            elif cmd in ["/help", "/?"]:
                render_help_menu()
                continue

            elif cmd in ["/stats", "/tokens"]:
                render_session_stats(agent)
                continue

            elif cmd in ["/tasks", "/all"]:
                render_tasks_table("view all")
                continue

            elif cmd == "/today":
                render_tasks_table("today | overdue")
                continue

            elif cmd.startswith("/done"):
                target = user_input[5:].strip()
                if not target:
                    console.print(
                        "[dim yellow]usage: /done <task name or ID>[/dim yellow]\n"
                    )
                    continue
                with console.status("[dim]completing task...[/dim]", spinner="dots"):
                    res = complete_task.invoke({"task_id_or_title": target})
                console.print(f"[dim green]✔ {res}[/dim green]\n")
                continue

            # ==================================================
            # 2. Windows File Drag-and-Drop Normalization
            # ==================================================
            # ปรับแต่ง Path กรณีผู้ใช้ลากไฟล์มาวางใน Terminal แล้วมีเครื่องหมายคำพูดครอบ
            if (user_input.startswith('"') and user_input.endswith('"')) or (user_input.startswith("'") and user_input.endswith("'")):
                candidate_path = user_input[1:-1].strip()
                if os.path.exists(candidate_path):
                    user_input = f"ช่วยตรวจสอบและอธิบายไฟล์นี้ให้หน่อย: {candidate_path}"

            # ==================================================
            # 3. Live Markdown Streaming with Card-based Tool Telemetry
            # ==================================================
            status = console.status(
                "[dim magenta]thinking...[/dim magenta]", spinner="dots"
            )
            status.start()

            live = None
            accumulated_text = ""

            try:
                for event_type, data in agent.chat_stream(user_input):

                    # Event: ได้รับ Token จาก LLM
                    if event_type == "token":
                        if status._live.is_started:
                            status.stop()

                        if live is None:
                            console.print()
                            live = Live(
                                Markdown(""),
                                console=console,
                                refresh_per_second=15,
                                vertical_overflow="visible",
                            )
                            live.start()

                        accumulated_text += data
                        live.update(Markdown(clean_markdown_text(accumulated_text)))


                    # Event: Agent สั่งรัน Tool
                    elif event_type == "tool_start":
                        func_name = data["name"]
                        func_args = data.get("args", {})
                        step_num = data.get("step", 1)

                        if live is not None:
                            live.stop()
                            live = None

                        # ล้าง accumulated_text ป้องกันข้อความซ้ำซ้อน
                        accumulated_text = ""

                        # ในโหมด Debug: แสดงรายละเอียด Step และ Arguments ด้วย Rich Card UI
                        if debug_mode:
                            if status._live.is_started:
                                status.stop()
                            
                            console.print(
                                f"\n[dim cyan]🔍 [Step {step_num}] Calling Tool: [bold white]{func_name}[/bold white][/dim cyan]"
                            )

                            # Card พิเศษสำหรับ execute_python: แสดง Syntax Highlighting
                            if func_name == "execute_python" and "code" in func_args:
                                code_content = func_args["code"]
                                syntax = Syntax(code_content, "python", theme="monokai", line_numbers=True)
                                panel = Panel(syntax, title="[bold yellow]🐍 Python Code to Execute[/bold yellow]", border_style="dim yellow", box=box.ROUNDED)
                                console.print(panel)
                            # Card พิเศษสำหรับ inspect_image: แสดง Path และรายละเอียด
                            elif func_name == "inspect_image":
                                img_path = func_args.get("image_path", "")
                                query_text = func_args.get("query", "")
                                console.print(f"[dim cyan]   ├─ Image: [white]{img_path}[/white][/dim cyan]")
                                console.print(f"[dim cyan]   └─ Query: [white]{query_text}[/white][/dim cyan]")
                            else:
                                console.print(
                                    f"[dim cyan]   └─ args: {func_args}[/dim cyan]"
                                )

                        status.update(f"[dim]executing {func_name}...[/dim]")
                        if not status._live.is_started:
                            status.start()

                    # Event: Tool ทำงานเสร็จ
                    elif event_type == "tool_end":
                        func_name = data["name"]
                        duration = data.get("duration", 0.0)
                        char_count = data.get("char_count", 0)

                        # ในโหมด Debug: แสดงเวลาที่ใช้และขนาดของผลลัพธ์
                        if debug_mode:
                            if status._live.is_started:
                                status.stop()
                            console.print(
                                f"[dim green]   ✔ finished {func_name} in {duration:.2f}s ({char_count:,} chars output)[/dim green]"
                            )

                        status.update("[dim magenta]thinking...[/dim magenta]")
                        if not status._live.is_started:
                            status.start()

                    # Event: สรุป Token Usage ประจำ Turn
                    elif event_type == "usage":
                        if status._live.is_started:
                            status.stop()
                        if live is not None:
                            live.update(Markdown(clean_markdown_text(accumulated_text)))

                            live.stop()
                            live = None

                        # ในโหมด Debug: แสดง Token Usage Bar
                        if debug_mode:
                            total_tok = data.get("total_tokens", 0)
                            in_tok = data.get("input_tokens", 0)
                            out_tok = data.get("output_tokens", 0)
                            speed = data.get("tokens_per_sec", 0.0)
                            dur = data.get("duration_sec", 0.0)
                            console.print(
                                f"\n[dim cyan]⚡ {total_tok:,} tokens ({in_tok:,} in / {out_tok:,} out) • {speed:.1f} tok/s • {dur:.2f}s turn[/dim cyan]"
                            )

                    # Event: เกิด Error
                    elif event_type == "error":
                        if status._live.is_started:
                            status.stop()
                        if live is not None:
                            live.stop()
                        console.print(f"\n[dim red]Error: {data}[/dim red]")

            finally:
                if status._live.is_started:
                    status.stop()
                if live is not None:
                    live.update(Markdown(clean_markdown_text(accumulated_text)))

                    live.stop()

            console.print()

        except KeyboardInterrupt:
            console.print("\n[dim]session cancelled.[/dim]")
            break
        except EOFError:
            break


if __name__ == "__main__":
    main()

