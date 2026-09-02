import os
import tempfile

from src.gaia_evaluator import GAIABenchmarkAgent


class TestGAIABenchmarkAgentHelpers:
    def test_create_empty_trace(self):
        agent = GAIABenchmarkAgent(max_steps=2, verbose=False)
        trace = agent._create_empty_trace()

        assert trace["tools_used"] == []
        assert trace["steps"] == 0
        assert trace["duration"] == 0.0
        assert trace["tokens"] == 0
        assert trace["speed"] == 0.0

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
