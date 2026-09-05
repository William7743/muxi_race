"""CPU-only preflight tests. Never load MACA libraries or access a GPU.

Run from any directory with Python's standard library only:
    python path/to/test_remote_mcpti_event_preflight.py
"""

import ctypes as C
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "remote_mcpti_event_preflight.py"
SPEC = importlib.util.spec_from_file_location("mcpti_event_preflight_under_test", SCRIPT)
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


class FakeFunction:
    """A callable accepting ctypes argtypes/restype without loading a library."""

    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


class EventPreflightTests(unittest.TestCase):
    def run_case(self, *, null_context=False, add_status=0, attr_status=0,
                 oversized=False, destroy_status=0, null_group=False):
        calls = []

        def mark(name, callback):
            def wrapped(*args):
                calls.append(name)
                return callback(*args)
            return FakeFunction(wrapped)

        def context(out):
            C.cast(out, C.POINTER(C.c_void_p)).contents.value = None if null_context else 1234
            return 0

        def version(out):
            C.cast(out, C.POINTER(C.c_uint32)).contents.value = 37
            return 0

        def create(ctx, out, flags):
            self.assertTrue(ctx.value)
            self.assertEqual(flags, 0)
            C.cast(out, C.POINTER(C.c_void_p)).contents.value = None if null_group else 5678
            return 0

        def attr(target, aid, out_size, out):
            size = C.cast(out_size, C.POINTER(C.c_size_t))
            self.assertEqual(size.contents.value, PREFLIGHT.RAW_CAPACITY)
            # Deliberately opaque fixture bytes: not a claimed MCPTI payload ABI.
            data = int(aid).to_bytes(4, "little")
            C.memmove(out, data, len(data))
            size.contents.value = PREFLIGHT.RAW_CAPACITY + 1 if oversized else len(data)
            return attr_status

        runtime = SimpleNamespace(
            mcSetDevice=mark("set_device", lambda dev: 0),
            mcCtxGetCurrent=mark("context", context),
        )
        pti = SimpleNamespace(
            mcptiGetVersion=mark("version", version),
            mcptiEventGetAttribute=mark("event_attr", attr),
            mcptiEventGroupCreate=mark("create", create),
            mcptiEventGroupDestroy=mark("destroy", lambda group: destroy_status),
            mcptiEventGroupAddEvent=mark("add", lambda group, event: add_status),
            mcptiEventGroupGetAttribute=mark("group_attr", attr),
        )
        # No Enable/Read/SetAttribute/Replay or kernel symbol is available in
        # these fakes: an accidental new call cannot silently pass the tests.
        report = PREFLIGHT.collect_metadata(
            runtime, pti, PREFLIGHT.new_report(0, 3, "/fake-not-a-library")
        )
        self.assertFalse(report["collection_enable_attempted"])
        self.assertFalse(report["collection_enabled"])
        self.assertFalse(report["profiling_scope_changed"])
        self.assertFalse(report["collection_mode_changed"])
        self.assertFalse(report["all_instances_changed"])
        self.assertFalse(report["replay_enabled"])
        self.assertIsNone(report["counter_values"])
        self.assertEqual(report["stage1_warmup_launches"], 0)
        self.assertEqual(report["stage1_collected_launches"], 0)
        return report, calls

    def test_success_is_metadata_not_collection(self):
        report, calls = self.run_case()
        self.assertEqual(report["status"], "metadata_only")
        self.assertEqual(calls[-1], "destroy")
        self.assertEqual(len(report["group_attributes"]), 6)
        self.assertTrue(all(item["typed_value"] is None
                            for item in report["group_attributes"]))

    def test_null_context_stops_before_create(self):
        report, calls = self.run_case(null_context=True)
        self.assertEqual(report["status"], "preflight_failed")
        self.assertNotIn("create", calls)

    def test_add_failure_still_destroys(self):
        report, calls = self.run_case(add_status=27)
        self.assertEqual(report["status"], "preflight_failed")
        self.assertEqual(calls[-1], "destroy")

    def test_attribute_failure_does_not_publish_buffer(self):
        report, calls = self.run_case(attr_status=11)
        self.assertEqual(report["status"], "metadata_partial")
        self.assertTrue(all(item["raw_hex"] is None for item in report["group_attributes"]))
        self.assertEqual(calls[-1], "destroy")

    def test_oversized_result_is_invalid(self):
        report, _ = self.run_case(oversized=True)
        self.assertEqual(report["status"], "metadata_partial")
        self.assertTrue(all(item["raw_hex"] is None for item in report["event_attributes"]))

    def test_cleanup_failure_not_silenced(self):
        report, _ = self.run_case(destroy_status=9)
        self.assertEqual(report["status"], "cleanup_failed")

    def test_null_group_not_destroyed(self):
        report, calls = self.run_case(null_group=True)
        self.assertEqual(report["status"], "preflight_failed")
        self.assertNotIn("destroy", calls)


if __name__ == "__main__":
    unittest.main(verbosity=2)
