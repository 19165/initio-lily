"""Interactive Terminal CLI for Lily AI Assistant.

This module provides a rich, minimalist command-line interface featuring:
- Live streaming markdown rendering via Rich Live.
- Interactive prompt with persistent history (.lily_history) and auto-completion.
- Built-in slash commands (/tasks, /today, /done, /debug, /clear, /exit).
- Debug & Tracing Mode (--debug or /debug) for tool telemetry and reasoning steps.
- Live PostgreSQL database status indicator.
"""

import sys

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
from rich.markdown import Markdown
from rich.text import Text
from rich.live import Live
from rich import box
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML

from src.agent import LilyAgent
from src.tools import get_tasks, complete_task

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
    "/tasks": "List all active tasks",
    "/today": "Show tasks due today / overdue",
    "/done": "Complete task by name or ID",
    "/debug": "Toggle debug / tracing mode",
    "/clear": "Clear screen",
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
# 3. Helper Functions
# -------------------------------------------------------------
def print_banner(agent: LilyAgent, debug_mode: bool = False):
    """แสดงโลโก้หัวโปรแกรม สถานะ Database และสถานะโหมด Debug."""
    logo = Text(LILY_ASCII_LOGO.strip("\n"), style="bold magenta")
    console.print(logo)

    db_badge = (
        "[bold green]● db online[/bold green]"
        if agent.db_available
        else "[bold yellow]○ db offline[/bold yellow]"
    )
    debug_badge = " • [bold cyan][DEBUG MODE][/bold cyan]" if debug_mode else ""

    console.print(
        f"[dim]lily v0.1 • local brain • {db_badge}{debug_badge} • type [cyan]/[/cyan] for commands[/dim]\n"
    )


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
            # 2. Live Markdown Streaming with Seamless Tool Execution
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
                        live.update(Markdown(accumulated_text))

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

                        # ในโหมด Debug: แสดงรายละเอียด Step และ Arguments
                        if debug_mode:
                            if status._live.is_started:
                                status.stop()
                            console.print(
                                f"\n[dim cyan]🔍 [Step {step_num}] Calling Tool: [bold]{func_name}[/bold][/dim cyan]"
                            )
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
                    live.update(Markdown(accumulated_text))
                    live.stop()

            console.print()

        except KeyboardInterrupt:
            console.print("\n[dim]session cancelled.[/dim]")
            break
        except EOFError:
            break


if __name__ == "__main__":
    main()
