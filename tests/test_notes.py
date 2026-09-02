import os
import shutil
import tempfile
from unittest.mock import patch

from src.tools.notes import (
    _get_safe_note_path,
    append_to_note,
    create_or_update_note,
    list_notes,
    read_note,
    search_notes,
)


class TestSafeNotePath:
    def test_sanitize_filename_and_add_extension(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch("src.tools.notes.NOTES_DIR", temp_dir):
            path = _get_safe_note_path("My: Note? *Ideas*")
            assert path.endswith("My Note Ideas.md")
            assert os.path.dirname(path) == os.path.abspath(temp_dir)

    def test_path_traversal_fails_or_sanitizes(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch("src.tools.notes.NOTES_DIR", temp_dir):
            # basename strips leading path traversal components
            path = _get_safe_note_path("../../etc/passwd")
            assert os.path.commonpath([os.path.abspath(temp_dir), path]) == os.path.abspath(temp_dir)


class TestNotesCRUDOperations:
    @classmethod
    def setup_class(cls):
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def teardown_class(cls):
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_create_and_read_note(self):
        with patch("src.tools.notes.NOTES_DIR", self.temp_dir):
            # 1. Create
            res = create_or_update_note.invoke({
                "title": "architecture_design",
                "content": "# Lily Architecture\nModular Layered Agent",
            })
            assert "Successfully saved note 'architecture_design.md'" in res

            # 2. Read
            content = read_note.invoke({"title": "architecture_design"})
            assert "# Lily Architecture" in content
            assert "Modular Layered Agent" in content

    def test_append_to_note(self):
        with patch("src.tools.notes.NOTES_DIR", self.temp_dir):
            # 1. Create initial
            create_or_update_note.invoke({
                "title": "daily_log",
                "content": "Log 1: System initialized.",
            })

            # 2. Append
            append_res = append_to_note.invoke({
                "title": "daily_log",
                "content": "Log 2: Local notes feature added.",
            })
            assert "Successfully appended to note 'daily_log.md'" in append_res

            # 3. Read back
            content = read_note.invoke({"title": "daily_log"})
            assert "Log 1: System initialized." in content
            assert "Log 2: Local notes feature added." in content

    def test_read_non_existent_note(self):
        with patch("src.tools.notes.NOTES_DIR", self.temp_dir):
            res = read_note.invoke({"title": "ghost_note_xyz_999"})
            assert "Error: Note 'ghost_note_xyz_999.md' not found." in res

    def test_list_notes(self):
        with patch("src.tools.notes.NOTES_DIR", self.temp_dir):
            res = list_notes.invoke({})
            assert "### 📝 Local Notes Knowledge Base:" in res
            assert "architecture_design.md" in res
            assert "daily_log.md" in res

    def test_search_notes(self):
        with patch("src.tools.notes.NOTES_DIR", self.temp_dir):
            # Search hit
            hit_res = search_notes.invoke({"query": "Modular Layered"})
            assert "### 🔍 Search Results for 'Modular Layered':" in hit_res
            assert "architecture_design.md" in hit_res
            assert "Modular Layered Agent" in hit_res

            # Search miss
            miss_res = search_notes.invoke({"query": "quantum_gravity_12345"})
            assert "No notes found matching 'quantum_gravity_12345'." in miss_res
