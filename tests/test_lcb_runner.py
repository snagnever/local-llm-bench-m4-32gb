import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bench2  # noqa: E402

ANNOTATED = """```python
class Solution:
    def total(self, nums: List[int]) -> int:
        return sum(nums)
```"""

FUNCTIONAL_Q = {
    "question_id": "q1",
    "testtype": "functional",
    "fn_name": "total",
    "starter_code": "class Solution:\n    def total(self, nums: List[int]) -> int:\n        ",
    "test_cases": [{"testtype": "functional", "input": "[1, 2, 3]", "output": "6"}],
}


def test_functional_runs_with_livecodebench_import_prelude():
    ok, _, reason = bench2.grade_livecodebench(FUNCTIONAL_Q, ANNOTATED)
    assert ok and reason == "pass", reason


def test_functional_input_larger_than_argmax():
    big = list(range(400_000))
    q = dict(FUNCTIONAL_Q, test_cases=[{"testtype": "functional", "input": json.dumps(big),
                                        "output": str(sum(big))}])
    ok, _, reason = bench2.grade_livecodebench(q, ANNOTATED)
    assert ok and reason == "pass", reason


def test_functional_wrong_answer_stays_wrong():
    q = dict(FUNCTIONAL_Q, test_cases=[{"testtype": "functional", "input": "[1, 2, 3]", "output": "7"}])
    ok, _, reason = bench2.grade_livecodebench(q, ANNOTATED)
    assert not ok and "mismatch" in reason


def test_stdin_runs_with_livecodebench_import_prelude():
    q = {"question_id": "q2", "testtype": "stdin", "starter_code": "",
         "test_cases": [{"testtype": "stdin", "input": "3\n", "output": "6"}]}
    code = "```python\nn = int(input())\nprint(sum(accumulate([1] * n)))\n```"
    ok, _, reason = bench2.grade_livecodebench(q, code)
    assert ok and reason == "pass", reason


def test_stdin_large_program_runs_from_file():
    # A program longer than ARG_MAX also fails with `python -c`.
    filler = "x = 0\n" * 300_000
    q = {"question_id": "q3", "testtype": "stdin", "starter_code": "",
         "test_cases": [{"testtype": "stdin", "input": "", "output": "ok"}]}
    ok, _, reason = bench2.grade_livecodebench(q, f"```python\n{filler}print('ok')\n```")
    assert ok and reason == "pass", reason


def test_temp_files_are_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(bench2.tempfile, "tempdir", str(tmp_path))
    bench2.grade_livecodebench(FUNCTIONAL_Q, ANNOTATED)
    assert list(tmp_path.iterdir()) == []
