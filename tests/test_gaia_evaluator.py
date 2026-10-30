"""Comprehensive test suite for GAIABenchmarkAgent and exact-match sanitizer."""

import os
import tempfile

from langchain_core.messages import AIMessage, BaseMessage

from src.gaia_evaluator import GAIABenchmarkAgent, sanitize_gaia_answer


class MockChatModel:
    """Deterministic mock chat model for GAIA benchmark unit tests."""

    def __init__(self, responses: list[AIMessage]):
        self.responses = list(responses)
        self.call_count = 0

    def bind_tools(self, tools: list[object]) -> "MockChatModel":
        return self

    def invoke(self, messages: list[BaseMessage], **kwargs: object) -> AIMessage:
        res = self.responses[self.call_count]
        if self.call_count < len(self.responses) - 1:
            self.call_count += 1
        return res


class TestGaiaAnswerSanitizer:
    """Unit tests for strict exact-match output sanitization."""

    def test_strip_markdown_bold_and_italics(self):
        assert sanitize_gaia_answer("**42**") == "42"
        assert sanitize_gaia_answer("*John Doe*") == "John Doe"
        assert sanitize_gaia_answer("***3.14***") == "3.14"
        assert sanitize_gaia_answer("`hello`") == "hello"

    def test_strip_conversational_prefixes(self):
        assert sanitize_gaia_answer("Final Answer: 42") == "42"
        assert sanitize_gaia_answer("The answer is: 100") == "100"
        assert sanitize_gaia_answer("Result: 99.5") == "99.5"
        assert sanitize_gaia_answer("Answer: **Paris**") == "Paris"

    def test_strip_enclosing_quotes(self):
        assert sanitize_gaia_answer('"42"') == "42"
        assert sanitize_gaia_answer("'Mount Everest'") == "Mount Everest"

    def test_strip_trailing_periods_safely(self):
        assert sanitize_gaia_answer("42.") == "42"
        assert sanitize_gaia_answer("-15.") == "-15"
        assert sanitize_gaia_answer("Completed.") == "Completed"

    def test_preserves_fractions_and_negatives(self):
        assert sanitize_gaia_answer("-42") == "-42"
        assert sanitize_gaia_answer("3/4") == "3/4"
        assert sanitize_gaia_answer("3.14159") == "3.14159"

    def test_empty_string(self):
        assert sanitize_gaia_answer("") == ""
        assert sanitize_gaia_answer("   ") == ""


class TestGAIABenchmarkAgentHelpers:
    """Tests for file resolution and trace creation."""

    def test_create_empty_trace(self):
        agent = GAIABenchmarkAgent(max_steps=2, verbose=False)
        trace = agent._create_empty_trace()

        assert trace["tools_used"] == []
        assert trace["steps"] == 0
        assert trace["duration"] == 0.0
        assert trace["tokens"] == 0
        assert trace["speed"] == 0.0
        assert trace["trajectory"] == []

    def test_resolve_file_with_direct_path(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("temporary gaia file")
            temp_path = f.name

        try:
            agent = GAIABenchmarkAgent(max_steps=2, verbose=False)
            resolved = agent._resolve_file(file_path=temp_path)
            assert resolved == temp_path
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_resolve_file_non_existent(self):
        agent = GAIABenchmarkAgent(max_steps=2, verbose=False)
        resolved = agent._resolve_file(file_name="completely_imaginary_gaia_file_xyz_999.pdf")
        assert resolved is None


class TestGAIABenchmarkAgentExecution:
    """Integration tests for benchmark execution with mock LLM."""

    def test_direct_answer_without_tools(self):
        mock_model = MockChatModel([
            AIMessage(
                content="Final Answer: **42**",
                usage_metadata={
                    "input_tokens": 20,
                    "output_tokens": 5,
                    "total_tokens": 25,
                },
            )
        ])
        evaluator = GAIABenchmarkAgent(
            max_steps=3,
            verbose=False,
            model=mock_model,
            use_memory_for_testing=True,
        )

        answer = evaluator("What is 40 + 2?", task_id="task_calc_1")

        assert answer == "42"
        trace = evaluator.last_trace
        assert trace["tools_used"] == []
        assert trace["steps"] == 1
        assert trace["tokens"] == 25
        assert trace["trajectory"] == []

    def test_react_tool_execution_and_trajectory(self):
        mock_model = MockChatModel([
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "python_interpreter",
                    "args": {"code": "print(21 * 2)"},
                    "id": "gaia_tool_call_1",
                }],
                usage_metadata={
                    "input_tokens": 15,
                    "output_tokens": 5,
                    "total_tokens": 20,
                },
            ),
            AIMessage(
                content="The answer is: **42**.",
                usage_metadata={
                    "input_tokens": 25,
                    "output_tokens": 5,
                    "total_tokens": 30,
                },
            ),
        ])
        evaluator = GAIABenchmarkAgent(
            max_steps=3,
            verbose=False,
            model=mock_model,
            use_memory_for_testing=True,
        )

        answer = evaluator("Calculate 21 * 2 using python", task_id="task_calc_2")

        assert answer == "42"
        trace = evaluator.last_trace
        assert trace["tools_used"] == ["python_interpreter"]
        assert trace["steps"] == 1
        assert trace["tokens"] == 50
        assert len(trace["trajectory"]) == 1

        step_record = trace["trajectory"][0]
        assert step_record["step"] == 1
        assert step_record["tool"] == "python_interpreter"
        assert step_record["args"] == {"code": "print(21 * 2)"}
        assert "42" in step_record["result_snippet"]
        assert step_record["duration"] >= 0

    def test_task_isolation_across_multiple_runs(self):
        mock_model = MockChatModel([
            AIMessage(content="Paris"),
            AIMessage(content="Tokyo"),
        ])
        evaluator = GAIABenchmarkAgent(
            max_steps=2,
            verbose=False,
            model=mock_model,
            use_memory_for_testing=True,
        )

        res1 = evaluator("Question 1", task_id="task_iso_1")
        res2 = evaluator("Question 2", task_id="task_iso_2")

        assert res1 == "Paris"
        assert res2 == "Tokyo"
