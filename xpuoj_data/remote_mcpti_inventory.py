"""Read-only MACA metric inventory; diagnostic only, never a submission import.

Declarations come from the installed MACA 3.7.1 mcpti.h / mcpti_type.h.
Initializes the chosen device context but does not configure counters, launch
kernels, enable replay, or change any submitted kernel. Enumerated metrics are
availability metadata, not collected performance measurements.
"""

import argparse
import ctypes as C
import json
from pathlib import Path


def bind(library, name, args):
    function = getattr(library, name)
    function.argtypes = args
    function.restype = C.c_int
    return function


def require_success(name, status):
    if status:
        raise RuntimeError(f"{name} returned status {status}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maca-root", type=Path, default=Path("/opt/maca"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--group-smoke", action="store_true",
                        help="also create/destroy an empty event group; never enable collection")
    args = parser.parse_args()
    runtime = C.CDLL(str(args.maca_root / "lib/libmcruntime.so"))
    set_device = bind(runtime, "mcSetDevice", [C.c_int])
    require_success("mcSetDevice", set_device(args.device))
    pti = C.CDLL(str(args.maca_root / "lib/libmcpti.so"))
    u32p, sizep = C.POINTER(C.c_uint32), C.POINTER(C.c_size_t)
    get_count = bind(pti, "mcptiDeviceGetNumMetrics", [C.c_int, u32p])
    enumerate_metrics = bind(pti, "mcptiDeviceEnumMetrics", [C.c_int, sizep, u32p])
    get_attr = bind(pti, "mcptiMetricGetAttribute", [C.c_uint32, C.c_int, sizep, C.c_void_p])
    get_event_count = bind(pti, "mcptiMetricGetNumEvents", [C.c_uint32, u32p])
    enumerate_events = bind(pti, "mcptiMetricEnumEvents", [C.c_uint32, sizep, u32p])
    count = C.c_uint32()
    require_success("mcptiDeviceGetNumMetrics", get_count(args.device, C.byref(count)))
    if count.value > 100000:
        raise RuntimeError("implausible metric count")
    ids = (C.c_uint32 * count.value)()
    size = C.c_size_t(C.sizeof(ids))
    if count.value:
        require_success("mcptiDeviceEnumMetrics", enumerate_metrics(args.device, C.byref(size), ids))
    if size.value > C.sizeof(ids) or size.value % C.sizeof(C.c_uint32):
        raise RuntimeError("invalid returned metric array size")

    def string_attribute(metric, attribute):
        buffer = C.create_string_buffer(16384)
        nbytes = C.c_size_t(C.sizeof(buffer))
        status = get_attr(metric, attribute, C.byref(nbytes), buffer)
        if status:
            return {"status": status}
        if b"\0" not in buffer.raw or nbytes.value > C.sizeof(buffer):
            raise RuntimeError(f"possibly truncated metric {metric} attribute {attribute}")
        return buffer.value.decode("utf-8", errors="replace")

    metrics = []
    for metric in ids[:size.value // C.sizeof(C.c_uint32)]:
        entry = {"id": metric, "name": string_attribute(metric, 0),
                 "description": string_attribute(metric, 1)}
        event_count = C.c_uint32()
        status = get_event_count(metric, C.byref(event_count))
        entry["event_count_status"] = status
        if not status:
            if event_count.value > 100000:
                raise RuntimeError("implausible event count")
            events = (C.c_uint32 * event_count.value)()
            event_bytes = C.c_size_t(C.sizeof(events))
            if event_count.value:
                require_success("mcptiMetricEnumEvents", enumerate_events(metric, C.byref(event_bytes), events))
            if event_bytes.value > C.sizeof(events) or event_bytes.value % C.sizeof(C.c_uint32):
                raise RuntimeError("invalid returned event array size")
            entry["event_ids"] = list(events[:event_bytes.value // C.sizeof(C.c_uint32)])
        metrics.append(entry)
    report = {"device": args.device, "maca_root": str(args.maca_root.resolve()),
              "reported_count": count.value, "metrics": metrics,
              "note": "Inventory only; no counters collected or replay enabled."}
    if args.group_smoke:
        current_context = bind(runtime, "mcCtxGetCurrent", [C.POINTER(C.c_void_p)])
        create_group = bind(pti, "mcptiEventGroupCreate",
                            [C.c_void_p, C.POINTER(C.c_void_p), C.c_uint32])
        destroy_group = bind(pti, "mcptiEventGroupDestroy", [C.c_void_p])
        context, group = C.c_void_p(), C.c_void_p()
        context_status = current_context(C.byref(context))
        smoke = {"context_status": context_status, "has_context": bool(context.value),
                 "create_status": None, "destroy_status": None, "collection_enabled": False}
        if context_status == 0 and context.value:
            status = create_group(context, C.byref(group), 0)
            smoke["create_status"] = status
            smoke["has_group"] = bool(group.value)
            if status == 0:
                smoke["destroy_status"] = destroy_group(group)
                require_success("mcptiEventGroupDestroy", smoke["destroy_status"])
        report["empty_group_smoke"] = smoke
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
