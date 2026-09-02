from unittest.mock import MagicMock

from src.utils import clean_markdown_text, format_tool_result, retry_on_network_error


class TestCleanMarkdownText:
    def test_empty_string(self):
        assert clean_markdown_text("") == ""

    def test_plain_text_unchanged(self):
        text = "Hello world! This is Lily 🌸"
        assert clean_markdown_text(text) == text

    def test_latex_arrow_replacement(self):
        text = "Step 1 $\\rightarrow$ Step 2 $\\to$ Step 3"
        result = clean_markdown_text(text)
        assert "→" in result
        assert "$\\rightarrow$" not in result

    def test_latex_math_symbols(self):
        text = "a \\times b \\approx c \\pm d"
        result = clean_markdown_text(text)
        assert "×" in result
        assert "≈" in result
        assert "±" in result

    def test_latex_sqrt_and_frac(self):
        text = "Value is \\sqrt{16} and ratio is \\frac{1}{2}"
        result = clean_markdown_text(text)
        assert "√(16)" in result
        assert "(1/2)" in result


class TestFormatToolResult:
    def test_none_input(self):
        assert format_tool_result(None) == "Success (No output returned)."

    def test_string_input_trimmed(self):
        assert format_tool_result("  hello world  ") == "hello world"

    def test_dict_input(self):
        data = {"name": "Lily", "role": "Assistant"}
        result = format_tool_result(data)
        assert '"name": "Lily"' in result
        assert '"role": "Assistant"' in result

    def test_exception_input(self):
        err = ValueError("Invalid parameter")
        result = format_tool_result(err)
        assert result == "Tool Error: Invalid parameter"

    def test_token_guard_truncation(self):
        huge_str = "x" * 5000
        result = format_tool_result(huge_str, max_chars=100)
        assert len(result) > 100
        assert "...[Truncated 4900 characters to protect context limit]..." in result
        assert result.startswith("x" * 100)


class TestRetryOnNetworkError:
    def test_successful_call_no_retry(self):
        mock_fn = MagicMock(return_value="OK")
        decorated = retry_on_network_error(max_retries=3, initial_delay=0.01)(mock_fn)
        assert decorated() == "OK"
        assert mock_fn.call_count == 1

    def test_retriable_glitch_recovers(self):
        mock_fn = MagicMock(side_effect=[ConnectionResetError("glitch"), "Recovered"])
        decorated = retry_on_network_error(max_retries=3, initial_delay=0.01)(mock_fn)
        assert decorated() == "Recovered"
        assert mock_fn.call_count == 2

    def test_retriable_glitch_exceeds_max_retries(self):
        mock_fn = MagicMock(side_effect=TimeoutError("timed out"))
        decorated = retry_on_network_error(max_retries=2, initial_delay=0.01)(mock_fn)
        res = decorated()
        assert "Network Error: Failed after 2 attempts" in res
        assert mock_fn.call_count == 2

    def test_non_retriable_error_does_not_retry(self):
        mock_fn = MagicMock(side_effect=ValueError("Bad value"))
        decorated = retry_on_network_error(max_retries=3, initial_delay=0.01)(mock_fn)
        res = decorated()
        assert "Error executing" in res
        assert "Bad value" in res
        assert mock_fn.call_count == 1
