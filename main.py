"""Interactive Terminal CLI for Lily AI Assistant.

This module provides a rich, minimalist command-line interface featuring:
- Live streaming markdown rendering via Rich Live.
- Interactive prompt with auto-completion using prompt_toolkit.
- Built-in slash commands (/tasks, /today, /done, /clear, /exit).
- Visual feedback for LLM tool execution and reasoning cycles.
"""

import sys

# -------------------------------------------------------------
# 0. Windows Console Encoding Fix
# -------------------------------------------------------------
# ป้องกันปัญหา UnicodeEncodeError (charmap codec) บน Windows เมื่อพิมพ์ข้อความภาษาไทยหรือ Emoji
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
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML

from src.agent import LilyAgent
from src.tools import get_tasks, complete_task

# Global Rich Console Instance สำหรับจัดการ Output บน Terminal
console = Console()

# -------------------------------------------------------------
# 2. UI Styling & Constants
# -------------------------------------------------------------
# โลโก้ ASCII สไตล์ Minimalist
LILY_ASCII_LOGO = r"""
   __ _ __     
  / /(_) /_ __ 
 / // / / // / 
/_//_/_/\_, /  
       /___/   
"""

# คำสั่งลัด (Slash Commands) พร้อมคำอธิบายสำหรับ Autocomplete Menu
SLASH_COMMANDS = {
    "/tasks": "List all active tasks",
    "/today": "Show tasks due today / overdue",
    "/done": "Complete task by name or ID",
    "/clear": "Clear screen",
    "/exit": "Exit assistant",
}

# กำหนดระบบ Autocompletion ให้แนะนำคำสั่งเมื่อผู้ใช้พิมพ์ '/'
slash_completer = WordCompleter(
    words=list(SLASH_COMMANDS.keys()),
    meta_dict=SLASH_COMMANDS,
    ignore_case=True,
    match_middle=True,
)

# ธีมสีสไตล์ Nord สำหรับกล่อง Prompt และ Autocomplete Popup
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
def print_banner():
    """แสดงโลโก้หัวโปรแกรมและเวอร์ชันของ Lily."""
    logo = Text(LILY_ASCII_LOGO.strip("\n"), style="bold magenta")
    console.print(logo)
    console.print(
        "[dim]lily v0.1 • local brain • type [cyan]/[/cyan] for commands[/dim]\n"
    )


def render_tasks_table(filter_query: str = ""):
    """ดึงข้อมูลจาก Todoist และแสดงผลเป็น Table ที่สวยงามบน Terminal.

    Args:
        filter_query: คำค้นหาฟิลเตอร์ เช่น 'view all', 'today | overdue'
    """
    # แสดง Spinner ระหว่างกำลังดึงข้อมูลจาก Todoist API
    with console.status("[dim]fetching tasks...[/dim]", spinner="dots"):
        raw_result = get_tasks.invoke({"filter_query": filter_query})

    # กรณีไม่มีรายการงานค้างอยู่
    if "No tasks found" in raw_result or not raw_result.strip():
        console.print("[dim yellow]No active tasks found.[/dim yellow]\n")
        return

    # สร้าง Table ด้วย Rich
    table = Table(
        box=box.ROUNDED,
        border_style="dim white",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("ID", style="dim", width=18)
    table.add_column("Task", style="white")
    table.add_column("Due", style="yellow")

    # แปลง Raw String จาก Tool ให้อยู่ในรูปแถวของตาราง
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
    # สร้าง Instance ของ Lily Agent (โหลด Database, Memory, Prompts และ Tools)
    agent = LilyAgent()

    # สร้าง Prompt Session พร้อมระบบ Autocomplete และ History
    session = PromptSession(completer=slash_completer, style=prompt_style)

    console.clear()
    print_banner()

    # ลูปรับคำสั่งจากผู้ใช้ไม่รู้จบ จนกว่าจะสั่ง /exit หรือกด Ctrl+C
    while True:
        try:
            # รับ Input จากผู้ใช้ผ่าน Prompt Symbol '❯'
            user_input = session.prompt(HTML("<ansicyan>❯ </ansicyan>")).strip()
            if not user_input:
                continue

            # ==================================================
            # 1. จัดการ Slash Commands (คำสั่งลัดที่จัดการฝั่ง Client โดยไม่ต้องผ่าน LLM)
            # ==================================================
            cmd = user_input.lower()

            # ออกจากโปรแกรม
            if cmd in ["/exit", "/quit", ":q"]:
                console.print("[dim magenta]bye! 🌸[/dim magenta]")
                break

            # เคลียร์หน้าจอ Terminal
            elif cmd == "/clear":
                console.clear()
                print_banner()
                continue

            # แสดงงานทั้งหมด
            elif cmd in ["/tasks", "/all"]:
                render_tasks_table("view all")
                continue

            # แสดงงานที่ครบกำหนดวันนี้ หรือเกินกำหนด
            elif cmd == "/today":
                render_tasks_table("today | overdue")
                continue

            # สั่งมาร์กงานเสร็จโดยตรง
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
            # เริ่มต้นแสดง Loading Status ระหว่างรอ Token แรกจากโมเดล
            status = console.status(
                "[dim magenta]thinking...[/dim magenta]", spinner="dots"
            )
            status.start()

            live = None
            accumulated_text = ""

            try:
                # วนลูปรับ Event จาก Agent Generator (Streaming Mode)
                for event_type, data in agent.chat_stream(user_input):

                    # กรณี Event: ได้รับ Token ข้อความจาก LLM
                    if event_type == "token":
                        # ปิด Spinner ทันทีที่เริ่มได้รับข้อความแรก
                        if status._live.is_started:
                            status.stop()

                        # เริ่มต้น Rich Live Renderer สำหรับการพิมพ์ข้อความแบบ Real-time
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

                    # กรณี Event: Agent สั่งรัน Tool (Function Calling)
                    elif event_type == "tool_start":
                        func_name = data["name"]

                        # หยุด Live Renderer เดิมชั่วคราว
                        if live is not None:
                            live.stop()
                            live = None

                        # ล้าง accumulated_text เพื่อป้องกันข้อความสะสมซ้ำซ้อนในลูปถัดไป
                        accumulated_text = ""

                        # เปลี่ยนสถานะแสดงชื่อ Tool ที่กำลังทำงานอยู่
                        status.update(f"[dim]executing {func_name}...[/dim]")
                        if not status._live.is_started:
                            status.start()

                    # กรณี Event: Tool ทำงานเสร็จสิ้น
                    elif event_type == "tool_end":
                        # สลับสถานะกลับมาเป็น 'thinking...' เพื่อรอ LLM สรุปผล
                        status.update("[dim magenta]thinking...[/dim magenta]")

                    # กรณี Event: เกิดข้อผิดพลาด
                    elif event_type == "error":
                        if status._live.is_started:
                            status.stop()
                        if live is not None:
                            live.stop()
                        console.print(f"\n[dim red]Error: {data}[/dim red]")

            finally:
                # Cleanup: ปิดสถานะและแสดงผลลัพธ์ Markdown ครั้งสุดท้ายให้เรียบร้อย
                if status._live.is_started:
                    status.stop()
                if live is not None:
                    live.update(Markdown(accumulated_text))
                    live.stop()

            console.print()

        except KeyboardInterrupt:
            # ผู้ใช้กด Ctrl+C เพื่อยกเลิก
            console.print("\n[dim]session cancelled.[/dim]")
            break
        except EOFError:
            # จบไฟล์หรือส่ง EOF
            break


if __name__ == "__main__":
    main()
