"""Offline tests: no OJ login, network access, or GPU required."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from xpuoj_feedback import main, summarize_submission


def result(time_us=9214):
    return {
        "time": time_us,
        "status": "Accepted",
        "displayScore": 79,
        "userError": json.dumps({
            "schema_version": 2, "time_ms": 9.214, "tk_time_ms": 9.214,
            "tb_time_ms": 36.025, "speedup": 3.909811, "pass": True,
        }),
        "checkerMessage": "baseline=35.875 ratio=0.795649",
    }


def detail():
    return {
        "meta": {"id": 139689, "status": "Accepted", "displayScore": 79.67},
        "content": {"code": "# XPU-OJ v713: test\r\n# XPU-OJ v496\r\n", "language": "tilelang.maca-c500"},
        "progress": {
            "status": "Accepted", "displayScore": 79.67,
            "samples": [{"testcaseHash": "sample"}],
            "subtasks": [
                {"score": 79, "fullScore": 100, "testcases": [{"testcaseHash": "case-b"}]},
                {"testcases": [{"testcaseHash": "case-a"}, {"testcaseHash": "case-c"}]},
            ],
            "testcaseResult": {
                "sample": result(999999), "case-c": result(), "case-a": result(2582),
                "unreferenced": result(1), "case-b": result(4658),
            },
        },
    }


class SummaryTests(unittest.TestCase):
    def test_sample_order_and_exact_units(self):
        summary = summarize_submission(detail())
        self.assertEqual(summary["sample_count"], 1)
        self.assertEqual(summary["formal_case_reference_count"], 3)
        self.assertEqual(summary["missing_case_result_count"], 0)
        self.assertEqual([case["testcaseHash"] for case in summary["cases"]], ["case-b", "case-a", "case-c"])
        self.assertEqual([case["time_us"] for case in summary["cases"]], [4658, 2582, 9214])
        self.assertEqual(summary["cases"][2]["telemetry"]["time_ms"], 9.214)
        self.assertEqual(summary["cases"][2]["telemetry"]["tb_time_ms"], 36.025)
        self.assertNotIn("checkerMessage", summary["cases"][2])

    def test_source_first_version_and_normalized_hash(self):
        payload = detail()
        summary = summarize_submission(payload)
        self.assertEqual(summary["source"]["version"], "v713")
        expected = hashlib.sha256(payload["content"]["code"].replace("\r\n", "\n").encode()).hexdigest()
        self.assertEqual(summary["source"]["sha256"], expected)
        payload["content"]["code"] = payload["content"]["code"].replace("\r\n", "\n")
        self.assertEqual(summarize_submission(payload)["source"]["sha256"], expected)

    def test_missing_result_never_infers_completion(self):
        payload = detail()
        payload["meta"]["status"] = "Running"
        payload["progress"]["status"] = "Running"
        del payload["progress"]["testcaseResult"]["case-a"]
        summary = summarize_submission(payload)
        self.assertEqual(summary["missing_case_result_count"], 1)
        self.assertFalse(summary["cases"][1]["result_available"])
        self.assertNotIn("score", summary["cases"][1])
        self.assertNotIn("time_us", summary["cases"][1])
        self.assertEqual(summary["meta"]["status"], "Running")
        self.assertNotIn("completed", summary)

    def test_absent_score_is_not_zero(self):
        summary = summarize_submission(detail())
        self.assertNotIn("score", summary["cases"][0])
        self.assertNotIn("score", summary["subtasks"][1])
        payload = detail()
        payload["progress"]["testcaseResult"]["case-b"]["score"] = 0
        self.assertEqual(summarize_submission(payload)["cases"][0]["score"], 0)

    def test_no_subtasks_does_not_use_all_results(self):
        payload = detail()
        del payload["progress"]["subtasks"]
        summary = summarize_submission(payload)
        self.assertEqual(summary["cases"], [])
        self.assertEqual(summary["formal_case_reference_count"], 0)
        self.assertEqual(summary["sample_count"], 1)
        self.assertNotIn("completed", summary)

    def test_bad_telemetry_is_not_echoed(self):
        for raw, expected in [("secret-not-json", "invalid_json"), ("[]", "not_object"), ('{"schema_version":1}', "unsupported_schema"), (None, "missing")]:
            with self.subTest(raw=raw):
                payload = detail()
                payload["progress"]["testcaseResult"]["case-b"]["userError"] = raw
                case = summarize_submission(payload)["cases"][0]
                self.assertEqual(case["telemetry_parse_status"], expected)
                self.assertNotIn("telemetry", case)
                self.assertNotIn("secret-not-json", json.dumps(case))

    def test_telemetry_only_finite_numeric_and_bool_fields(self):
        payload = detail()
        payload["progress"]["testcaseResult"]["case-b"]["userError"] = '{"schema_version":2,"time_ms":true,"tk_time_ms":"9","tb_time_ms":NaN,"speedup":3.9,"pass":"true","token":"secret"}'
        telemetry = summarize_submission(payload)["cases"][0]["telemetry"]
        self.assertEqual(telemetry, {"schema_version": 2, "speedup": 3.9})

    def test_secrets_and_raw_payloads_excluded(self):
        payload = detail()
        secrets = {"token": "secret-token", "password": "secret-password", "email": "private@example.test", "progressSubscriptionKey": "secret-subscription"}
        for location in (payload, payload["meta"], payload["progress"], payload["content"], payload["progress"]["testcaseResult"]["case-b"]):
            location.update(secrets)
        payload["progress"]["testcaseResult"]["case-b"]["userOutput"] = "OJCHAL secret-challenge OJRESULT raw-output"
        payload["progress"]["testcaseResult"]["case-b"]["input"] = "secret-input"
        payload["content"]["compileAndRunOptions"] = secrets
        payload["content"]["code"] += "# private-source-marker\n"
        rendered = json.dumps(summarize_submission(payload))
        for value in [*secrets.keys(), *secrets.values(), "userOutput", "OJCHAL", "raw-output", "secret-input", "private-source-marker", "compileAndRunOptions"]:
            self.assertNotIn(value, rendered)

    def test_checker_is_explicit_opt_in(self):
        report = summarize_submission(detail(), include_checker=True)
        self.assertEqual(report["cases"][0]["checkerMessage"], "baseline=35.875 ratio=0.795649")

    def test_invalid_reference_stays_missing(self):
        payload = detail()
        payload["progress"]["subtasks"][0]["testcases"] = [{}]
        summary = summarize_submission(payload)
        self.assertFalse(summary["cases"][0]["result_available"])
        self.assertEqual(summary["missing_case_result_count"], 1)

    def test_top_level_decoy_is_never_used_or_merged(self):
        payload = detail()
        payload["testcaseResult"] = {"case-b": result(1), "case-a": result(2)}
        summary = summarize_submission(payload)
        self.assertEqual(summary["cases"][0]["time_us"], 4658)
        self.assertEqual(summary["cases"][1]["time_us"], 2582)

        del payload["progress"]["testcaseResult"]["case-a"]
        summary = summarize_submission(payload)
        self.assertFalse(summary["cases"][1]["result_available"])
        self.assertNotIn("time_us", summary["cases"][1])

        del payload["progress"]["testcaseResult"]
        summary = summarize_submission(payload)
        self.assertEqual(summary["missing_case_result_count"], 3)
        self.assertTrue(all(not case["result_available"] for case in summary["cases"]))


class CliTests(unittest.TestCase):
    def fake_client_module(self):
        module = types.ModuleType("xpuoj_submit")
        module.load_credentials = mock.Mock(return_value=("private@example.test", "secret-password"))
        client = mock.Mock()
        client.get_submission_detail.return_value = detail()
        module.XPUOJClient = mock.Mock(return_value=client)
        return module, client

    def test_one_detail_only_and_exclusive_report(self):
        module, client = self.fake_client_module()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            output = io.StringIO()
            with mock.patch.dict("sys.modules", {"xpuoj_submit": module}), contextlib.redirect_stdout(output):
                self.assertEqual(main(["--detail", "139689", "--output", str(path)]), 0)
            client.get_submission_detail.assert_called_once_with(139689)
            client.submit.assert_not_called()
            client.wait_for_result.assert_not_called()
            client.session.close.assert_called_once()
            self.assertEqual(json.loads(output.getvalue()), json.loads(path.read_text(encoding="utf-8")))
            before = path.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--detail", "139689", "--output", str(path)])
            self.assertEqual(path.read_bytes(), before)
            self.assertNotIn("private@example.test", output.getvalue())

    def test_client_error_body_not_echoed(self):
        module, client = self.fake_client_module()
        client.get_submission_detail.side_effect = RuntimeError("token=secret-password private@example.test")
        output = io.StringIO()
        with mock.patch.dict("sys.modules", {"xpuoj_submit": module}), contextlib.redirect_stderr(output):
            self.assertEqual(main(["--detail", "139689"]), 1)
        self.assertNotIn("secret-password", output.getvalue())
        self.assertNotIn("private@example.test", output.getvalue())
        self.assertIn("RuntimeError", output.getvalue())


if __name__ == "__main__":
    unittest.main()
