"""Parse TENNOR console timings without choosing a measurement window.

Only a ``Training Batch ... takes ... seconds`` line completes a batch.  The
reported total and the cumulative global timer are preserved independently.
Stage durations follow the explicitly agreed layer-1 boundaries; timestamps
have one-second resolution and readback duration fields are already seconds.
No source code, datasets, or third-party Python packages are needed.
"""

from collections import defaultdict
from datetime import datetime
import math
import re


_MONTHS = {month: index for index, month in enumerate(
    "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), 1)}
_CTIME = re.compile(
    r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Za-z]{3})\s+(\d{1,2})\s+"
    r"(\d{2}):(\d{2}):(\d{2})\s+(\d{4})$")
_EPOCH = re.compile(r"^Epoch\s*:?\s*(\d+)\s*:?")
_COMPLETE = re.compile(r"^Training Batch\s+(\d+)\s+takes\s+(.+?)\s+seconds$")
_GLOBAL = re.compile(
    r"^Global training time after Batch\s+(\d+)\s+takes\s+(.+?)\s+seconds$")
_INIT = re.compile(r"^Network Initialization Duration:\s*(.+?)\s+seconds$")
_FF = re.compile(r"^Start FF:\s*layer\s+1\s*,\s*(.+)$")
_READBACK_HEADER = "LSH->Requester(readback) time breakdown (sec):"
_READBACK_FIELD = re.compile(r"^(scheduler|parallel|free|oht\.clear|total)\s*:\s*(.*?)\s*$")
_EVENT_NAMES = {
    "Start Update": "update",
    "Start FF": "ff",
    "Start LSH Rehash update": "rehash",
    "Start Requester sort by ID": "sort",
    "End Requester Merge": "merge_end",
    "Start Neuron ADAM update": "adam_start",
    "End Neuron ADAM update": "adam_end",
    "Start O3 fast OHT rebuild": "rebuild",
    "Start access by LSH": "readback_start",
    "End write Requester back to LSH": "readback_end",
}
_FF_NAMES = {
    "compute Request Buc IDs": "request",
    "build OHT": "oht",
    "OHT access by LSH": "lsh",
    "OHT access by Requester": "reqtrans",
    "compute activation": "activation",
}


def _number(raw):
    """Return a finite nonnegative duration; never turn malformed data into zero."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _timestamp(raw):
    """Parse English ctime text independently of the machine's current locale."""
    match = _CTIME.fullmatch(raw)
    if not match or match[1] not in _MONTHS:
        return None
    try:
        return datetime(int(match[6]), _MONTHS[match[1]], int(match[2]),
                        int(match[3]), int(match[4]), int(match[5]))
    except ValueError:
        return None


def _iso(value):
    return value.isoformat(sep=" ") if value is not None else None


def _sum(*values):
    return sum(values) if all(value is not None for value in values) else None


def _parse_batch(lines, start, end, epoch, batch, occurrence, segment, reported, audit, warnings):
    identity = {"epoch": epoch, "batch": batch, "occurrence": occurrence, "segment": segment}
    row = dict(identity, header_line=start + 1, end_line=end + 1,
               reported_seconds=_number(reported), global_seconds=None,
               start_time=None, errors=[])
    errors = row["errors"]

    def problem(message):
        if message not in errors:
            errors.append(message)

    if row["reported_seconds"] is None:
        problem(f"line {end + 1}: invalid reported batch duration {reported!r}")

    occurrences = defaultdict(list)
    for index in range(start, end):
        label = " ".join(lines[index].split())
        name = _EVENT_NAMES.get(label)
        match = _FF.fullmatch(label)
        if match:
            name = _FF_NAMES.get(match[1])
        if name:
            occurrences[name].append(index)

    cache = {}

    def event(name, required=True):
        if name in cache:
            result, error = cache[name]
        else:
            indices = occurrences[name]
            result, error = None, None
            if len(indices) != 1:
                error = (f"missing event {name!r}" if not indices else
                         f"ambiguous repeated event {name!r} at lines "
                         + ", ".join(str(index + 1) for index in indices))
            else:
                index = indices[0]
                next_index = index + 1
                while next_index < end and not lines[next_index].strip():
                    next_index += 1
                raw = lines[next_index].strip() if next_index < end else ""
                parsed = _timestamp(raw)
                result = {"line": index + 1, "timestamp_line": next_index + 1,
                          "time": parsed, "raw": raw}
                if parsed is None:
                    error = f"line {index + 1}: missing or malformed timestamp for {name!r}: {raw!r}"
            cache[name] = (result, error)
        if error:
            if required:
                problem(error)
            else:
                warnings.append(f"batch {batch}, line {start + 1}: {error} (diagnostic only)")
        return result

    def interval(name, first, last, required=True):
        begin, finish = event(first, required), event(last, required)
        error, seconds = None, None
        if begin is None or finish is None or begin["time"] is None or finish["time"] is None:
            error = f"{name}: missing or invalid interval boundary"
        elif begin["line"] >= finish["line"]:
            error = f"{name}: interval boundaries are out of order"
        else:
            seconds = (finish["time"] - begin["time"]).total_seconds()
            if seconds < 0:
                error = f"{name}: negative timestamp duration {seconds} seconds"
                seconds = None
        if error:
            if required:
                problem(error)
            else:
                warnings.append(f"batch {batch}, line {start + 1}: {error} (diagnostic only)")
        audit.append(dict(identity, kind="interval", component=name,
                          start_line=begin["line"] if begin else None,
                          end_line=finish["line"] if finish else None,
                          start_timestamp_line=begin["timestamp_line"] if begin else None,
                          end_timestamp_line=finish["timestamp_line"] if finish else None,
                          start_time=_iso(begin["time"]) if begin else None,
                          end_time=_iso(finish["time"]) if finish else None,
                          seconds=seconds, valid=error is None, error=error))
        return seconds

    start_event = event("update")
    row["start_time"] = _iso(start_event["time"]) if start_event else None
    requester_prepare = interval("requester_prepare", "request", "oht")
    requester_merge = interval("requester_merge", "sort", "merge_end")
    oht_forward = interval("oht_forward", "oht", "lsh")
    oht_rebuild = interval("oht_rebuild", "rebuild", "readback_start")
    lsh_forward = interval("lsh_forward", "lsh", "reqtrans")
    reqtrans = interval("requester_transfer", "reqtrans", "activation")
    updater_bp = interval("updater_backprop", "activation", "sort")
    updater_adam = interval("updater_adam", "adam_start", "adam_end")
    row["rehash_present"] = bool(occurrences["rehash"])
    row["rehash_s"] = interval("rehash", "rehash", "ff") if row["rehash_present"] else 0.0
    row["rehash_valid"] = row["rehash_s"] is not None
    row["readback_wall_s"] = interval("readback_wall", "readback_start", "readback_end", required=False)

    # The intervals form a sequential partition.  Checking all shared boundaries
    # also catches overlaps across otherwise individually well-formed intervals.
    order = ["update"]
    if row["rehash_present"]:
        order.extend(["rehash", "ff"])
    order.extend(["request", "oht", "lsh", "reqtrans", "activation", "sort",
                  "merge_end", "adam_start", "adam_end", "rebuild", "readback_start"])
    for earlier, later in zip(order, order[1:]):
        before, after = event(earlier), event(later)
        if before and after:
            invalid = before["line"] >= after["line"]
            if before["time"] is not None and after["time"] is not None:
                invalid = invalid or before["time"] > after["time"]
            if invalid:
                problem(f"stage order or overlapping scope: {earlier!r} must precede {later!r}")
                if earlier in ("update", "rehash", "ff") or later in ("rehash", "ff"):
                    row["rehash_valid"] = False

    headers = [index for index in range(start, end)
               if lines[index].strip() == _READBACK_HEADER]
    fields = {}
    if len(headers) != 1:
        problem("missing readback duration block" if not headers else "ambiguous repeated readback duration blocks")
    else:
        header = headers[0]
        readback_start = event("readback_start")
        if readback_start and header + 1 <= readback_start["line"]:
            problem("readback duration block precedes Start access by LSH")
        end_indices = occurrences["readback_end"]
        if len(end_indices) == 1 and header >= end_indices[0]:
            problem("readback duration block follows End write Requester back to LSH")
        index = header + 1
        while index < end:
            raw = lines[index].strip()
            if not raw:
                index += 1
                continue
            match = _READBACK_FIELD.fullmatch(raw)
            if not match:
                break
            key, raw_value = match.groups()
            value = _number(raw_value)
            duplicate = key in fields
            if duplicate:
                problem(f"line {index + 1}: repeated readback field {key!r}")
                fields[key] = None
            else:
                fields[key] = value
            if value is None:
                problem(f"line {index + 1}: invalid readback duration {key}={raw_value!r}")
            audit.append(dict(identity, kind="readback_field", component=key,
                              source_line=index + 1, raw=raw_value, seconds=value,
                              valid=value is not None and not duplicate))
            index += 1
    for key in ("total", "oht.clear"):
        if key not in fields:
            problem(f"missing readback duration {key!r}")
    total, clear = fields.get("total"), fields.get("oht.clear")
    lsh_readback = total - clear if total is not None and clear is not None else None
    if lsh_readback is not None and lsh_readback < 0:
        problem("negative readback LSH duration: total is smaller than oht.clear")
        lsh_readback = None

    row.update(requester_s=_sum(requester_prepare, requester_merge),
               oht_s=_sum(oht_forward, oht_rebuild, clear),
               lsh_scan_s=_sum(lsh_forward, lsh_readback), reqtrans_s=reqtrans,
               updater_s=_sum(updater_bp, updater_adam),
               readback_total_s=total, readback_clear_s=clear)
    row["fetcher_s"] = _sum(row["oht_s"], row["lsh_scan_s"], row["reqtrans_s"])
    row["breakdown_valid"] = not errors
    warnings.extend(f"batch {batch}, line {start + 1}: {error}" for error in errors)
    return row


def parse_console(text):
    """Return ``batches``, ``audit``, ``initializations``, and ``warnings`` lists.

    ``batches`` contains the latest complete usable occurrence of each global
    batch number, ordered by its location in the input.  TENNOR batch numbering
    continues across epochs: a repeated number is a retry, even when its epoch
    label changes.  Epoch labels are metadata, never part of batch identity.
    Without an explicit epoch header, epoch is None.  Missing
    or invalid measurements remain None, and a complete batch whose component
    data is unusable remains present with ``breakdown_valid=False``.  An invalid
    reported total cannot be used as a measurement and is audited and excluded.
    ``segment`` is a zero-based execution identifier.  Explicit initialization,
    incomplete-block restarts, or non-increasing global batch numbers start a
    new segment regardless of epoch labels.  Cumulative timers must not be
    differenced across segment boundaries, including after deduplication.

    Audit entries preserve 1-based source lines for intervals, raw numeric
    readback fields, discarded incomplete blocks, and superseded duplicates.
    Initialization durations are separate and never added to a batch total.
    """
    lines = text.splitlines()
    batches, audit, initializations, warnings = [], [], [], []
    epoch, active, last_completed = None, None, None
    occurrences = defaultdict(int)
    segment, execution_started = 0, False
    previous_batch = None
    last_unlabelled = None

    def discard(end, reason):
        if active is not None:
            audit.append({"kind": "incomplete_batch", "epoch": active[1],
                          "batch": None, "header_line": active[0] + 1,
                          "end_line": end, "segment": active[2],
                          "action": "discarded", "reason": reason})
            warnings.append(f"line {active[0] + 1}: discarded incomplete batch ({reason})")

    def restart(source_line, reason):
        nonlocal segment, previous_batch
        segment += 1
        previous_batch = None
        audit.append({"kind": "execution_boundary", "source_line": source_line,
                      "segment": segment, "reason": reason})
        warnings.append(f"line {source_line}: execution segment {segment} starts ({reason})")

    for index, raw in enumerate(lines):
        label = raw.strip()
        if label == "Start Init Network":
            discard(index, "network initialization before completion")
            if execution_started:
                restart(index + 1, "new network initialization")
                epoch = None
            execution_started = True
            active, last_completed = None, None
            continue
        match = _INIT.fullmatch(label)
        if match:
            seconds = _number(match[1])
            initializations.append({"line": index + 1, "source_line": index + 1,
                                    "seconds": seconds, "raw": match[1], "valid": seconds is not None})
            if seconds is None:
                warnings.append(f"line {index + 1}: invalid network initialization duration {match[1]!r}")
            continue
        match = _EPOCH.fullmatch(label)
        if match:
            discard(index, "new epoch header before completion")
            if active is not None:
                restart(index + 1, "new epoch header after an incomplete batch")
            elif epoch is not None and int(match[1]) < epoch:
                restart(index + 1, "explicit epoch number moved backwards")
            active, last_completed = None, None
            epoch = int(match[1])
            last_unlabelled = None
            continue
        if label == "Start Update":
            discard(index, "next Start Update before completion")
            if active is not None:
                restart(index + 1, "new Start Update after an incomplete batch")
            active, last_completed = (index, epoch, segment), None
            execution_started = True
            continue
        match = _COMPLETE.fullmatch(label)
        if match:
            number = int(match[1])
            if active is None:
                audit.append({"kind": "orphan_completion", "source_line": index + 1,
                              "epoch": epoch, "batch": number, "action": "discarded"})
                warnings.append(f"line {index + 1}: completion has no preceding Start Update")
                continue
            block_start, block_epoch, block_segment = active
            key = number
            occurrences[key] += 1
            if block_epoch is None:
                if last_unlabelled is not None and number <= last_unlabelled["batch"]:
                    audit.append({"kind": "unlabelled_batch_reset", "epoch": None,
                                  "source_line": index + 1, "header_line": block_start + 1,
                                  "previous_source_line": last_unlabelled["source_line"],
                                  "previous_header_line": last_unlabelled["header_line"],
                                  "old_batch": last_unlabelled["batch"], "new_batch": number,
                                  "previous_segment": last_unlabelled["segment"],
                                  "reason": "non-increasing batch number without explicit epoch labels"})
                    warnings.append(f"line {index + 1}: batch numbering reset without an Epoch header; "
                                    "global batch numbers identify retries; keeping the latest complete "
                                    "occurrence with a usable total")
            if previous_batch is not None and number <= previous_batch:
                restart(block_start + 1, "non-increasing global batch number indicates a retry")
                block_segment = segment
            previous_batch = number
            if block_epoch is None:
                last_unlabelled = {"batch": number, "source_line": index + 1,
                                   "header_line": block_start + 1, "segment": block_segment}
            if _number(match[2]) is None:
                audit.append({"kind": "invalid_completion", "source_line": index + 1,
                              "header_line": block_start + 1, "end_line": index + 1,
                              "epoch": block_epoch, "batch": number,
                              "occurrence": occurrences[key], "segment": block_segment,
                              "raw": match[2], "action": "discarded",
                              "reason": "invalid reported batch duration"})
                warnings.append(f"line {index + 1}: discarded invalid reported batch duration {match[2]!r}")
                active, last_completed = None, None
                continue
            row = _parse_batch(lines, block_start, index, block_epoch, number,
                               occurrences[key], block_segment, match[2], audit, warnings)
            batches.append(row)
            active, last_completed = None, row
            continue
        match = _GLOBAL.fullmatch(label)
        if match:
            number = int(match[1])
            if last_completed is None or last_completed["batch"] != number:
                warnings.append(f"line {index + 1}: unmatched global training duration for batch {number}")
            elif last_completed.get("global_line") is not None:
                warnings.append(f"line {index + 1}: repeated global training duration for batch {number}; "
                                "keeping the first value")
            else:
                last_completed["global_seconds"] = _number(match[2])
                last_completed["global_line"] = index + 1
                if last_completed["global_seconds"] is None:
                    warnings.append(f"line {index + 1}: invalid global training duration {match[2]!r}")
    discard(len(lines), "end of file before completion")

    latest = {}
    for row in batches:
        key = row["batch"]
        if key in latest:
            prior = latest[key]
            audit.append({"kind": "duplicate_batch", "epoch": prior["epoch"],
                          "batch": prior["batch"], "occurrence": prior["occurrence"],
                          "segment": prior["segment"], "kept_segment": row["segment"],
                          "kept_epoch": row["epoch"],
                          "header_line": prior["header_line"], "end_line": prior["end_line"],
                          "action": "discarded", "reason": "superseded by latest complete occurrence",
                          "kept_header_line": row["header_line"], "kept_occurrence": row["occurrence"]})
            warnings.append(f"batch {row['batch']}: kept complete occurrence {row['occurrence']} "
                            f"at line {row['header_line']}; discarded occurrence {prior['occurrence']}")
        latest[key] = row
    selected = sorted(latest.values(), key=lambda row: row["header_line"])
    return {"batches": selected, "audit": audit,
            "initializations": initializations, "warnings": warnings}
