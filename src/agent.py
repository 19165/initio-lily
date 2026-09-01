from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from src.config import OLLAMA_MODEL, OLLAMA_BASE_URL, SYSTEM_PROMPT
from src.tools import ALL_TOOLS


class LilyAgent:
    def __init__(self, max_steps: int = 5):
        self.llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        self.tools = ALL_TOOLS
        self.tools_by_name = {t.name: t for t in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.messages = [SystemMessage(content=SYSTEM_PROMPT)]
        self.max_steps = max_steps

    def chat(self, user_input: str) -> str:
        """Run an autonomous ReAct loop with multi-step reasoning and dynamic tool execution."""
        self.messages.append(HumanMessage(content=user_input))

        for _ in range(self.max_steps):
            response = self.llm_with_tools.invoke(self.messages)
            self.messages.append(response)

            # If no tool calls, return final response
            if not response.tool_calls:
                return response.content

            # Dynamic Dispatching for tool calls
            for call in response.tool_calls:
                tool_name = call["name"]
                tool_args = call["args"]
                call_id = call["id"]

                tool = self.tools_by_name.get(tool_name)
                if tool:
                    result = tool.invoke(tool_args)
                else:
                    result = f"Error: Tool '{tool_name}' not found."

                self.messages.append(
                    ToolMessage(content=str(result), tool_call_id=call_id)
                )

        return "ขออภัยค่ะ ลิลลี่คิดขั้นตอนวนซ้ำเกินกำหนด (Max steps exceeded)"
