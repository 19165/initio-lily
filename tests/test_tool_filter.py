from src.tools.filter import filter_tools_by_intent


class TestToolFilteringByIntent:
    def test_search_intent_selects_web_tools(self):
        tools = filter_tools_by_intent("ผมอยากให้เธอ web search ทั้ง 7 ท่านั่นต่างหาก")
        tool_names = {t.name for t in tools}
        assert "search_web" in tool_names
        assert "fetch_web_page" in tool_names
        assert len(tools) <= 7

    def test_notes_intent_selects_notes_tools(self):
        tools = filter_tools_by_intent("ช่วยจดโน้ตสรุปแผนงานประจำสัปดาห์นี้ให้หน่อย")
        tool_names = {t.name for t in tools}
        assert "create_or_update_note" in tool_names
        assert "read_note" in tool_names
        assert "append_to_note" in tool_names
        assert "list_notes" in tool_names

    def test_tasks_intent_selects_todoist_tools(self):
        tools = filter_tools_by_intent("มี task งานอะไรต้องทำบ้างวันนี้")
        tool_names = {t.name for t in tools}
        assert "get_tasks" in tool_names
        assert "add_task" in tool_names
        assert "complete_task" in tool_names

    def test_file_code_intent_selects_benchmark_tools(self):
        tools = filter_tools_by_intent("อ่านเอกสาร financial_report.pdf ให้หน่อย")
        tool_names = {t.name for t in tools}
        assert "read_document" in tool_names

    def test_general_chat_returns_compact_base_tools(self):
        tools = filter_tools_by_intent("สวัสดีตอนเช้า สบายดีไหม")
        tool_names = {t.name for t in tools}
        assert "get_current_time" in tool_names
        assert "save_fact" in tool_names
        # Ensure we don't dump all 17 tools for basic chit-chat
        assert len(tools) <= 7
