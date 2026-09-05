#!/usr/bin/env python3
"""Read one OJ detail response and emit a small, allowlisted feedback report.

The pure summarize_submission function never accesses the network. The CLI
only logs in and reads one submission; it does not submit, poll, or save secrets.
"""

import argparse
from collections.abc import Mapping
import getpass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace


_META_KEYS = ("id", "status", "displayScore", "submitTime", "timeUsed")
_SCORE_KEYS = ("score", "fullScore", "displayScore")
_TELEMETRY_NUMBERS = ("time_ms", "tk_time_ms", "tb_time_ms", "speedup")
_VERSION = re.compile(r"\bXPU-OJ\s+v(\d+)\b", re.IGNORECASE)


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def _number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and (not isinstance(value, float) or math.isfinite(value))
    )


def _fields(value, names):
    """Copy scalar allowlisted fields, never recursive payload objects."""
    result = {}
    for name in names:
        if name not in value:
            continue
        item = value[name]
        if item is None or isinstance(item, (str, bool)) or _number(item):
            result[name] = item
    return result


def _telemetry(user_error):
    if user_error is None or user_error == "":
        return {}, "missing"
    if not isinstance(user_error, str):
        return {}, "not_json_string"
    try:
        parsed = json.loads(user_error)
    except (ValueError, TypeError):
        return {}, "invalid_json"
    if not isinstance(parsed, dict):
        return {}, "not_object"
    if parsed.get("schema_version") != 2:
        return {}, "unsupported_schema"
    result = {"schema_version": 2}
    for name in _TELEMETRY_NUMBERS:
        if _number(parsed.get(name)):
            result[name] = parsed[name]
    if isinstance(parsed.get("pass"), bool):
        result["pass"] = parsed["pass"]
    return result, "parsed"


def summarize_submission(payload, *, include_checker=False):
    """Summarize only explicitly referenced scoring cases, in their given order.

    Missing fields stay missing: an absent score is not zero. Missing case
    results are represented explicitly. No completion/acceptance is inferred
    from scores, result counts, or the presence of telemetry. Platform statuses
    remain raw scalar metadata. Samples and unreferenced results are excluded.
    Source code, userOutput, input data, credential fields, and subscription
    keys are never copied. Checker text is opt-in and is treated as report data.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("Submission detail must be an object")
    meta = _mapping(payload.get("meta"))
    content = _mapping(payload.get("content"))
    progress = _mapping(payload.get("progress"))
    # The verified API nests results under progress. Never fall back to a
    # similarly named top-level field, which is not the scoring result map.
    results = _mapping(progress.get("testcaseResult"))
    source = _fields(content, ("language",))
    code = content.get("code")
    if isinstance(code, str):
        normalized = code.replace("\r\n", "\n").replace("\r", "\n")
        source["sha256"] = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        version = _VERSION.search(normalized)
        if version:
            source["version"] = "v" + version.group(1)

    samples = progress.get("samples")
    subtasks = progress.get("subtasks")
    if not isinstance(subtasks, list):
        subtasks = []
    summary = {
        "meta": _fields(meta, _META_KEYS),
        "source": source,
        "progress": _fields(progress, ("status", "displayScore")),
        "sample_count": len(samples) if isinstance(samples, list) else 0,
        "formal_case_reference_count": 0,
        "missing_case_result_count": 0,
        "subtasks": [],
        "cases": [],
    }
    for subtask_index, raw_subtask in enumerate(subtasks, start=1):
        subtask = _mapping(raw_subtask)
        references = subtask.get("testcases")
        if not isinstance(references, list):
            references = []
        summary["subtasks"].append({
            "subtask_index": subtask_index,
            **_fields(subtask, _SCORE_KEYS),
            "case_reference_count": len(references),
        })
        for case_index, reference in enumerate(references, start=1):
            case = {"subtask_index": subtask_index, "case_index": case_index}
            testcase_hash = _mapping(reference).get("testcaseHash")
            result = None
            if isinstance(testcase_hash, str):
                case["testcaseHash"] = testcase_hash
                result = results.get(testcase_hash)
            case["result_available"] = isinstance(result, Mapping)
            summary["formal_case_reference_count"] += 1
            if not case["result_available"]:
                summary["missing_case_result_count"] += 1
            else:
                case.update(_fields(result, ("score", "displayScore", "status")))
                if _number(result.get("time")):
                    # The platform's testcaseResult.time is already microseconds.
                    case["time_us"] = result["time"]
                telemetry, parse_status = _telemetry(result.get("userError"))
                case["telemetry_parse_status"] = parse_status
                if telemetry:
                    case["telemetry"] = telemetry
                if include_checker and isinstance(result.get("checkerMessage"), str):
                    case["checkerMessage"] = result["checkerMessage"]
            summary["cases"].append(case)
    return summary


def _credentials(loader):
    try:
        return loader(SimpleNamespace(email=None, password=None))
    except SystemExit:
        if not sys.stdin.isatty():
            raise RuntimeError("Credentials required for non-interactive query") from None
        # Hidden input for both fields; neither is persisted or printed.
        return getpass.getpass("OJ login email: "), getpass.getpass("OJ password: ")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail", type=int, required=True, help="Submission ID to read once")
    parser.add_argument("--include-checker", action="store_true", help="Include raw SPJ report text")
    parser.add_argument("--output", type=Path, help="Create a new JSON report; never overwrite")
    args = parser.parse_args(argv)
    if args.detail <= 0:
        parser.error("--detail must be positive")
    if args.output and args.output.exists():
        parser.error("--output already exists; choose a new report path")

    try:
        # Lazy import keeps offline summaries/tests independent of requests.
        from xpuoj_submit import XPUOJClient, load_credentials

        email, password = _credentials(load_credentials)
        client = XPUOJClient(email, password)
        try:
            payload = client.get_submission_detail(args.detail)
        finally:
            client.session.close()
        report = summarize_submission(payload, include_checker=args.include_checker)
        rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(rendered)
        print(rendered, end="")
        return 0
    except Exception as error:
        # Existing client exceptions may include raw API bodies. Do not echo them.
        print(f"Feedback query failed ({type(error).__name__}); no raw response printed.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
