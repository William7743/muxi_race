"""Metadata-only MCPTI event preflight; this is NOT a kernel collector.

This version NEVER enables counters, changes collection mode/scope, enables
replay, allocates model tensors, compiles a candidate, or launches a kernel.
It creates one disabled event group, attempts to add one inventory event,
reads bounded raw attribute buffers, and destroys the group. Group attribute
payload types/writability are not documented in the available header copy;
raw bytes are deliberately NOT interpreted as context-scoped collection.

ABI: MACA 3.7.1 mcpti.h and mcpti_type.h downloaded 2026-09-05. This diagnostic
must never be imported by an OJ submission. Run standalone in the coordinated
diagnostic window; enumeration/create success does not prove collection works.
"""

import argparse
import ctypes as C
import json
import sys
from pathlib import Path


HEADER_SHA256 = {
    "mcpti.h": "8d280365099be08c332c2960fb7ef71314274746acfad799c001b94229fc8b1b",
    "mcpti_type.h": "a9be1b75203e1cee64b2f453fcb6e2f5496fb188116f0d55e510e86ab7e32f69",
}
# Attribute IDs come from the downloaded header, not CUPTI assumptions.
EVENT_ATTRIBUTES = ((0, "name"), (1, "short_description"),
                    (2, "long_description"), (3, "category"),
                    (5, "profiling_scope"))
GROUP_ATTRIBUTES = ((0, "event_domain_id"), (1, "profile_all_domain_instances"),
                    (3, "num_events"), (4, "events"),
                    (5, "instance_count"), (6, "profiling_scope"))
RAW_CAPACITY = 4096


class PreflightError(RuntimeError):
    """Safe stop before any counter enable or kernel launch."""


def bind(library, name, args):
    function = getattr(library, name)
    function.argtypes = args
    function.restype = C.c_int
    return function


def raw_attribute(function, target, attribute, name):
    """Read bounded opaque bytes; do not infer a uint32 payload from its size."""
    buffer = (C.c_ubyte * RAW_CAPACITY)(*([0xA5] * RAW_CAPACITY))
    nbytes = C.c_size_t(RAW_CAPACITY)
    status = function(target, attribute, C.byref(nbytes), buffer)
    result = {
        "attribute_id": attribute,
        "attribute_name": name,
        "status": status,
        "capacity_bytes": RAW_CAPACITY,
        "returned_size_bytes": nbytes.value,
        "raw_hex": None,
        "typed_value": None,
        "payload_type_verified": False,
    }
    if status == 0:
        if nbytes.value == 0 or nbytes.value > RAW_CAPACITY:
            result["validation_error"] = "invalid_returned_size"
        else:
            result["raw_hex"] = bytes(buffer[:nbytes.value]).hex()
    return result


def new_report(device, event_id, maca_root):
    return {
        "schema_version": 1,
        "purpose": "metadata_only_preflight_for_future_stage1_smoke",
        "device": device,
        "event_id": event_id,
        "maca_root": str(maca_root),
        "abi_header_sha256": HEADER_SHA256,
        "status": "not_started",
        "api_status": {},
        "event_attributes": [],
        "group_attributes": [],
        "collection_enable_attempted": False,
        "collection_enabled": False,
        "collection_mode_changed": False,
        "profiling_scope_changed": False,
        "all_instances_changed": False,
        "replay_enabled": False,
        "model_tensors_allocated": False,
        "candidate_compiled": False,
        "stage1_warmup_launches": 0,
        "stage1_collected_launches": 0,
        "counter_values": None,
        "context_scope_verified": False,
        "cleanup": {"destroy_attempted": False, "destroy_status": None},
    }


def collect_metadata(runtime, pti, report):
    """All GPU-facing dependencies are injected for offline testing."""
    u32p = C.POINTER(C.c_uint32)
    sizep = C.POINTER(C.c_size_t)
    voidpp = C.POINTER(C.c_void_p)
    context, group = C.c_void_p(), C.c_void_p()
    created = False

    def checked_status(name, status):
        report["api_status"][name] = status
        if status != 0:
            raise PreflightError(f"{name} returned status {status}")

    try:
        # Bind all required symbols before creating a resource. In particular,
        # destroy must exist before create can be attempted.
        set_device = bind(runtime, "mcSetDevice", [C.c_int])
        get_context = bind(runtime, "mcCtxGetCurrent", [voidpp])
        get_version = bind(pti, "mcptiGetVersion", [u32p])
        event_attr = bind(pti, "mcptiEventGetAttribute",
                          [C.c_uint32, C.c_int, sizep, C.c_void_p])
        create = bind(pti, "mcptiEventGroupCreate",
                      [C.c_void_p, voidpp, C.c_uint32])
        destroy = bind(pti, "mcptiEventGroupDestroy", [C.c_void_p])
        add_event = bind(pti, "mcptiEventGroupAddEvent", [C.c_void_p, C.c_uint32])
        group_attr = bind(pti, "mcptiEventGroupGetAttribute",
                          [C.c_void_p, C.c_int, sizep, C.c_void_p])

        checked_status("mcSetDevice", set_device(report["device"]))
        version = C.c_uint32()
        checked_status("mcptiGetVersion", get_version(C.byref(version)))
        report["mcpti_api_version"] = version.value
        checked_status("mcCtxGetCurrent", get_context(C.byref(context)))
        report["has_context"] = bool(context.value)
        if not context.value:
            raise PreflightError("current context is null; no implicit kernel launch attempted")

        report["event_attributes"] = [
            raw_attribute(event_attr, report["event_id"], attribute, name)
            for attribute, name in EVENT_ATTRIBUTES
        ]
        checked_status("mcptiEventGroupCreate", create(context, C.byref(group), 0))
        report["has_group"] = bool(group.value)
        if not group.value:
            raise PreflightError("group-create returned success with null handle")
        created = True
        checked_status("mcptiEventGroupAddEvent", add_event(group, report["event_id"]))
        report["group_attributes"] = [
            raw_attribute(group_attr, group, attribute, name)
            for attribute, name in GROUP_ATTRIBUTES
        ]
        failed_attrs = [
            {"source": source, "attribute_id": item["attribute_id"]}
            for source in ("event_attributes", "group_attributes")
            for item in report[source]
            if item["status"] != 0 or "validation_error" in item
        ]
        report["failed_attributes"] = failed_attrs
        report["status"] = "metadata_partial" if failed_attrs else "metadata_only"
        report["stop_reason"] = (
            "group_attribute_payload_type_and_context_scope_contract_not_verified; "
            "no_enable_or_stage1_launch_permitted"
        )
    except Exception as exc:
        report["status"] = "preflight_failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        # No Enable symbol is bound or called by this program; consequently
        # Disable is neither necessary nor attempted. Destroy the disabled
        # group even when add/query failed. Never reset the device.
        if created:
            report["cleanup"]["destroy_attempted"] = True
            try:
                status = destroy(group)
                report["cleanup"]["destroy_status"] = status
                if status != 0:
                    report["status"] = "cleanup_failed"
            except Exception as exc:
                report["status"] = "cleanup_failed"
                report["cleanup"]["error"] = f"{type(exc).__name__}: {exc}"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maca-root", type=Path, default=Path("/opt/maca-3.7.1"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--event-id", type=int, choices=(3, 9, 10, 69), default=3)
    args = parser.parse_args()
    if args.device < 0:
        parser.error("device must be nonnegative")
    report = new_report(args.device, args.event_id, args.maca_root.resolve())
    try:
        runtime = C.CDLL(str(args.maca_root / "lib/libmcruntime.so"))
        pti = C.CDLL(str(args.maca_root / "lib/libmcpti.so"))
        collect_metadata(runtime, pti, report)
    except Exception as exc:
        report["status"] = "preflight_failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(report, indent=2))
    # metadata_only is a successful PRE-FLIGHT, never a successful collection.
    return 0 if report["status"] == "metadata_only" else 1


if __name__ == "__main__":
    sys.exit(main())
