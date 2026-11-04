import base64
import io
from unittest.mock import MagicMock

import pytest
from PIL import Image

from src.utils import (
    build_multimodal_human_message,
    calculate_jaccard_similarity,
    clean_markdown_text,
    encode_image_to_base64,
    format_tool_result,
    is_image_file,
    retry_on_network_error,
)


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
        res = str(decorated())
        assert "Network Error: Failed after 2 attempts" in res
        assert mock_fn.call_count == 2

    def test_non_retriable_error_does_not_retry(self):
        mock_fn = MagicMock(side_effect=ValueError("Bad value"))
        decorated = retry_on_network_error(max_retries=3, initial_delay=0.01)(mock_fn)
        res = str(decorated())
        assert "Error executing" in res
        assert "Bad value" in res
        assert mock_fn.call_count == 1


class TestCalculateJaccardSimilarity:
    def test_identical_queries(self):
        assert calculate_jaccard_similarity("hello world", "hello world") == 1.0

    def test_completely_different_queries(self):
        assert calculate_jaccard_similarity("apple orange", "car bus airplane") == 0.0

    def test_case_insensitive_and_word_order_insensitivity(self):
        sim = calculate_jaccard_similarity("Malko winners list 1980", "1980 list winners malko")
        assert sim == 1.0

    def test_near_duplicate_high_threshold(self):
        # 5 out of 6 words overlap -> 5/6 = 0.833 >= 0.75
        sim = calculate_jaccard_similarity(
            "Malko Competition 1980 first prize winner",
            "Malko Competition 1980 first prize"
        )
        assert sim >= 0.75

    def test_empty_or_none_inputs(self):
        assert calculate_jaccard_similarity("", "test") == 0.0
        assert calculate_jaccard_similarity("test", "") == 0.0
        assert calculate_jaccard_similarity("", "") == 0.0


class TestMultimodalUtils:
    @pytest.fixture
    def sample_image_bytes(self):
        buf = io.BytesIO()
        img = Image.new("RGB", (200, 100), color="pink")
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_is_image_file_valid_extensions(self):
        assert is_image_file("diagram.PNG") is True
        assert is_image_file("photo.jpg") is True
        assert is_image_file("image.jpeg") is True
        assert is_image_file("chart.webp") is True
        assert is_image_file("drawing.bmp") is True
        assert is_image_file("animation.gif") is True
        assert is_image_file("scan.tiff") is True

    def test_is_image_file_invalid_extensions(self):
        assert is_image_file("document.pdf") is False
        assert is_image_file("table.xlsx") is False
        assert is_image_file("script.py") is False
        assert is_image_file("notes.txt") is False
        assert is_image_file("") is False
        assert is_image_file(None) is False

    def test_encode_image_to_base64_from_bytes(self, sample_image_bytes):
        encoded = encode_image_to_base64(sample_image_bytes)
        assert isinstance(encoded, str)
        assert len(encoded) > 0
        decoded_bytes = base64.b64decode(encoded)
        decoded_img = Image.open(io.BytesIO(decoded_bytes))
        assert decoded_img.size == (200, 100)

    def test_encode_image_to_base64_from_file(self, tmp_path):
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (100, 50), color="blue")
        img.save(img_path)

        encoded = encode_image_to_base64(str(img_path))
        assert isinstance(encoded, str)
        decoded_bytes = base64.b64decode(encoded)
        decoded_img = Image.open(io.BytesIO(decoded_bytes))
        assert decoded_img.size == (100, 50)

    def test_encode_image_proportional_resize(self):
        # Create oversized image: 2000 x 1000
        buf = io.BytesIO()
        img = Image.new("RGB", (2000, 1000), color="green")
        img.save(buf, format="PNG")

        encoded = encode_image_to_base64(buf.getvalue(), max_dimension=500)
        decoded_img = Image.open(io.BytesIO(base64.b64decode(encoded)))
        # Expect scaled to 500 x 250
        assert decoded_img.size == (500, 250)

    def test_encode_image_invalid_source_raises(self):
        with pytest.raises(FileNotFoundError):
            encode_image_to_base64("non_existent_file.png")

        with pytest.raises(ValueError):
            encode_image_to_base64(b"not an image byte stream")

        with pytest.raises(ValueError):
            encode_image_to_base64("")

    def test_build_multimodal_human_message_text_only(self):
        msg = build_multimodal_human_message("Hello Lily")
        assert msg.content == "Hello Lily"

    def test_build_multimodal_human_message_with_image(self, sample_image_bytes):
        msg = build_multimodal_human_message("Look at this", images=[sample_image_bytes])
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2
        assert msg.content[0] == {"type": "text", "text": "Look at this"}
        assert msg.content[1]["type"] == "image_url"
        assert msg.content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_build_multimodal_human_message_image_only(self, sample_image_bytes):
        msg = build_multimodal_human_message("", images=[sample_image_bytes])
        assert isinstance(msg.content, list)
        assert len(msg.content) == 1
        assert msg.content[0]["type"] == "image_url"


