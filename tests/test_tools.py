import os
import tempfile
import pytest
from datetime import datetime
from src.tools import get_current_time, read_document, execute_python


class TestSystemTools:
    def test_get_current_time_format(self):
        result = get_current_time.invoke({})
        # Format is %d-%m-%Y, %H:%M:%S
        parsed = datetime.strptime(result, "%d-%m-%Y, %H:%M:%S")
        assert parsed is not None


class TestReadDocumentTool:
    def test_file_not_found(self):
        res = read_document.invoke({"file_path": "non_existent_file_xyz_123.txt"})
        assert "Error: File not found" in res

    def test_read_txt_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Lily is the best AI assistant!")
            path = f.name

        try:
            res = read_document.invoke({"file_path": path})
            assert "Lily is the best AI assistant!" in res
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_unsupported_format(self):
        with tempfile.NamedTemporaryFile("w", suffix=".unsupported_ext", delete=False, encoding="utf-8") as f:
            f.write("Some data")
            path = f.name

        try:
            res = read_document.invoke({"file_path": path})
            assert "Error: Unsupported file format '.unsupported_ext'" in res
        finally:
            if os.path.exists(path):
                os.remove(path)


class TestExecutePythonTool:
    def test_execute_valid_python(self):
        code = "print(2 + 2)"
        res = execute_python.invoke({"code": code})
        assert res.strip() == "4"

    def test_execute_no_print_output(self):
        code = "x = 42"
        res = execute_python.invoke({"code": code})
        assert "Did you forget to use print()?" in res

    def test_execute_runtime_error(self):
        code = "1 / 0"
        res = execute_python.invoke({"code": code})
        assert "ZeroDivisionError" in res
