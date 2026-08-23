from shopping_grpo.evaluation.artifacts import (
    append_jsonl_fsync,
    write_json_atomic,
    write_jsonl_atomic,
)


def test_json_artifacts_are_utf8_with_lf_on_every_platform(tmp_path):
    jsonl_path = tmp_path / "rows.jsonl"
    json_path = tmp_path / "manifest.json"

    write_jsonl_atomic(jsonl_path, [{"b": 2, "a": 1}])
    append_jsonl_fsync(jsonl_path, {"message": "澄清"})
    write_json_atomic(json_path, {"b": 2, "a": 1})

    assert jsonl_path.read_bytes() == (
        b'{"a": 1, "b": 2}\n'
        + '{"message": "澄清"}\n'.encode("utf-8")
    )
    assert json_path.read_bytes() == b'{\n  "a": 1,\n  "b": 2\n}\n'
    assert b"\r\n" not in jsonl_path.read_bytes()
    assert b"\r\n" not in json_path.read_bytes()
