"""Small JSON/JSONL artifact helpers with fail-closed overwrite semantics."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile


class ArtifactError(ValueError):
    """Raised for malformed, duplicate, or unsafe evaluation artifacts."""


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ArtifactError(
                    f"{source}:{line_number}: invalid JSON"
                ) from exc
            if not isinstance(row, dict):
                raise ArtifactError(
                    f"{source}:{line_number}: JSONL row must be an object"
                )
            yield row


def load_json(path: str | Path) -> dict:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{source}: invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ArtifactError(f"{source}: JSON root must be an object")
    return payload


def index_jsonl(
    path: str | Path,
    *,
    key: str,
    allowed_keys: set[object] | None = None,
) -> dict[object, dict]:
    result = {}
    for row_number, row in enumerate(iter_jsonl(path), start=1):
        if key not in row:
            raise ArtifactError(f"{path}:{row_number}: missing key {key!r}")
        value = row[key]
        if allowed_keys is not None and value not in allowed_keys:
            raise ArtifactError(
                f"{path}:{row_number}: unexpected {key}={value!r}"
            )
        if value in result:
            raise ArtifactError(
                f"{path}:{row_number}: duplicate {key}={value!r}"
            )
        result[value] = row
    return result


def load_unique_task_ids(path: str | Path) -> set[int]:
    """Load normalized, unique integer task IDs from a JSONL artifact."""
    task_ids: set[int] = set()
    for row_number, row in enumerate(iter_jsonl(path), start=1):
        if "task_id" not in row:
            raise ArtifactError(f"{path}:{row_number}: missing key 'task_id'")
        try:
            task_id = int(row["task_id"])
        except (TypeError, ValueError) as exc:
            raise ArtifactError(
                f"{path}:{row_number}: task_id must be an integer"
            ) from exc
        if task_id in task_ids:
            raise ArtifactError(f"{path}:{row_number}: duplicate task_id={task_id}")
        task_ids.add(task_id)
    if not task_ids:
        raise ArtifactError(f"{path}: no task IDs")
    return task_ids


def guard_task_id_disjoint(
    candidate_paths: Iterable[str | Path], evaluation_path: str | Path
) -> dict:
    """Fail closed when any candidate training task occurs in final evaluation."""
    evaluation_ids = load_unique_task_ids(evaluation_path)
    candidate_ids: set[int] = set()
    candidate_rows = 0
    for path in candidate_paths:
        ids = load_unique_task_ids(path)
        candidate_rows += len(ids)
        candidate_ids.update(ids)
    overlap = sorted(candidate_ids & evaluation_ids)
    if overlap:
        raise ArtifactError(
            "training data overlaps final evaluation task IDs: "
            f"{overlap[:20]} (total={len(overlap)})"
        )
    return {
        "evaluation_tasks": str(Path(evaluation_path).resolve()),
        "evaluation_rows": len(evaluation_ids),
        "candidate_rows": candidate_rows,
        "unique_candidate_tasks": len(candidate_ids),
        "overlap_count": 0,
    }


def _prepare_output(path: str | Path, *, force: bool) -> Path:
    output = Path(path)
    if output.exists() and not force:
        raise FileExistsError(
            f"output already exists: {output}; pass force=True to replace it"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


@contextmanager
def atomic_jsonl_writer(
    path: str | Path,
    *,
    force: bool = False,
) -> Iterator[Callable[[Mapping], None]]:
    """Yield a row writer and publish only after the whole stream succeeds."""

    output = _prepare_output(path, force=force)
    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(temporary.name)

    def write_row(row: Mapping) -> None:
        if not isinstance(row, Mapping):
            raise ArtifactError("JSONL output row must be an object")
        temporary.write(
            json.dumps(
                dict(row),
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            + "\n"
        )

    try:
        with temporary:
            yield write_row
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output)
    except BaseException:
        try:
            temporary.close()
        finally:
            temporary_path.unlink(missing_ok=True)
        raise


def write_jsonl_atomic(
    path: str | Path,
    rows: Iterable[Mapping],
    *,
    force: bool = False,
) -> None:
    with atomic_jsonl_writer(path, force=force) as write:
        for row in rows:
            write(row)


def append_jsonl_fsync(path: str | Path, row: Mapping) -> None:
    """Durably append one validated remote-model result for resumable runs."""

    if not isinstance(row, Mapping):
        raise ArtifactError("JSONL output row must be an object")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(
                dict(row),
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def write_json_atomic(
    path: str | Path,
    payload: Mapping,
    *,
    force: bool = False,
) -> None:
    output = _prepare_output(path, force=force)
    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            json.dump(
                dict(payload),
                temporary,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=str,
            )
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output)
    except BaseException:
        try:
            temporary.close()
        finally:
            temporary_path.unlink(missing_ok=True)
        raise
