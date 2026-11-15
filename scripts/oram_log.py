"""Parse timestamped SLIDE-ORAM batches without using opaque C++ timers.

A batch is measurable only when the next consecutive batch header arrives in
the same execution and exactly ``batchsize`` FF/BP/Free sequences are present.
Initialization, termination, retries, and skipped headers break this pairing.
All timestamps are the launcher's local, second-resolution receipt times.
"""

from collections import defaultdict
from datetime import datetime
import re


_PREFIX = re.compile(r"^\[([^\]]*)\]\s*(.*)$")
_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
_BATCH = re.compile(r"^START TRAINING Batch\s+(\d+)$")
_LSH = re.compile(r"^LSH update Start,\s*node id\s*:\s*(\d+)$")
_SAMPLE = {"FeedForward Start": "ff", "BackPropagation Start": "bp", "Free Memory": "free"}


def _line(raw, number):
    match = _PREFIX.fullmatch(raw.strip())
    label, stamp, value = raw.strip(), None, None
    if match:
        stamp, label = match.groups()
        if _STAMP.fullmatch(stamp):
            try:
                value = datetime.fromisoformat(stamp)
            except ValueError:
                pass
    return {"line": number, "label": label.strip(), "raw_timestamp": stamp, "time": value}


def _iso(value):
    return value.isoformat(sep=" ") if value is not None else None


def _execution_boundary(label):
    """Recognize actual process boundaries, including the recorded typo Starrt."""
    message = re.sub(r"^\[launcher\]\s*", "", label, flags=re.IGNORECASE)
    if re.search(r"\b(?:network initialization|init network)\b", message, re.IGNORECASE):
        return "network initialization"
    if re.search(r"\bStarting\s+runme\b", message, re.IGNORECASE):
        return "launcher start"
    if re.match(r"Run\s+status\s*:", message, re.IGNORECASE):
        return "launcher termination status"
    if re.match(r"(?:Restart(?:ing)?|Process\s+(?:terminated|exited)|Run\s+(?:terminated|interrupted|failed|completed|finished))\b",
                message, re.IGNORECASE):
        return "execution restart or termination"
    return None


def _identity(attempt):
    return {key: attempt[key] for key in ("batch", "occurrence", "segment")}


def _finish(attempt, end, batchsize, audit, warnings):
    """Reject corrupt sample structure; retain valid totals with invalid rehash."""
    start, events = attempt["start"], attempt["events"]
    identity = _identity(attempt)
    errors = []
    samples = [event for event in events if event["label"] in _SAMPLE]
    updates = [event for event in events if event["label"] == "Update Start"]
    lsh = [event for event in events if event["label"].startswith("LSH update Start")]
    need_rehash = [event for event in events if event["label"] == "Need Rehash"]
    counts = {kind: sum(_SAMPLE[event["label"]] == kind for event in samples)
              for kind in ("ff", "bp", "free")}
    first_ff = next((event for event in samples if event["label"] == "FeedForward Start"), None)
    for kind, count in counts.items():
        if count != batchsize:
            errors.append(f"expected {batchsize} {kind} events, found {count}")
    sequence = [_SAMPLE[event["label"]] for event in samples]
    if sequence != [kind for _ in range(batchsize) for kind in ("ff", "bp", "free")]:
        errors.append("sample event order must be exactly FF, BP, Free for each sample")
    if len(updates) != 1:
        errors.append(f"expected one Update Start, found {len(updates)}")
    update = updates[0] if len(updates) == 1 else None
    if update and first_ff and update["line"] >= first_ff["line"]:
        errors.append("Update Start must precede the first FeedForward Start")
    if lsh and (first_ff is None or any(event["line"] >= first_ff["line"] for event in lsh)):
        errors.append("LSH update events after sample processing starts have an unknown scope")
    if lsh and update and any(event["line"] <= update["line"] for event in lsh):
        errors.append("LSH update events must follow Update Start")

    # Validate all duration and sample boundaries.  LSH-only timestamp damage
    # invalidates the refresh measurement separately, preserving a valid total.
    core = [start] + [event for event in events
                     if event["label"] == "Update Start" or event["label"] in _SAMPLE] + [end]
    previous = None
    for event in core:
        if event["time"] is None:
            errors.append(f"line {event['line']}: missing or malformed timestamp for {event['label']!r}")
        else:
            if previous is not None and event["time"] < previous["time"]:
                errors.append(f"line {event['line']}: timestamp regressed after line {previous['line']}")
            previous = event
    seconds = None
    if start["time"] is not None and end["time"] is not None:
        seconds = (end["time"] - start["time"]).total_seconds()
        if seconds < 0:
            errors.append("negative batch duration between consecutive batch headers")
            seconds = None
    if errors:
        audit.append(dict(identity, kind="invalid_batch", header_line=start["line"],
                          end_line=end["line"], action="discarded", reasons=errors,
                          ff_count=counts["ff"], bp_count=counts["bp"], free_count=counts["free"]))
        warnings.append(f"batch {attempt['batch']}, line {start['line']}: discarded invalid batch: "
                        + "; ".join(errors))
        return None

    update_seconds = (first_ff["time"] - update["time"]).total_seconds()
    audit.append(dict(identity, kind="interval", component="update", start_line=update["line"],
                      end_line=first_ff["line"], start_time=_iso(update["time"]),
                      end_time=_iso(first_ff["time"]), seconds=update_seconds, valid=True,
                      note="enclosing update diagnostic; not added to rehash or batch time"))
    rehash_errors, rehash_events = [], []
    rehash_seconds = 0.0
    if lsh:
        previous = update
        previous_node = None
        for progress_index, event in enumerate(lsh):
            progress = _LSH.fullmatch(event["label"])
            if not progress:
                rehash_errors.append(f"line {event['line']}: malformed LSH node progress event")
            else:
                node = int(progress[1])
                if progress_index == 0 and node != 0:
                    rehash_errors.append(f"line {event['line']}: rehash must start at node 0, found node {node}")
                if previous_node is not None and node <= previous_node:
                    rehash_errors.append(f"line {event['line']}: non-increasing LSH node progress; "
                                         "cannot infer separate complete rehash events")
                previous_node = node
            if event["time"] is None:
                rehash_errors.append(f"line {event['line']}: missing or malformed LSH timestamp")
            else:
                if previous["time"] is not None and event["time"] < previous["time"]:
                    rehash_errors.append(f"line {event['line']}: LSH timestamp regressed after line {previous['line']}")
                if event["time"] > first_ff["time"]:
                    rehash_errors.append(f"line {event['line']}: LSH timestamp exceeds first FeedForward boundary")
                previous = event
        first = lsh[0]
        rehash_seconds = ((first_ff["time"] - first["time"]).total_seconds()
                          if first["time"] is not None and not rehash_errors else None)
        if rehash_seconds is not None and rehash_seconds < 0:
            rehash_errors.append("negative rehash interval")
            rehash_seconds = None
        measured = dict(identity, kind="interval", component="rehash", start_line=first["line"],
                        end_line=first_ff["line"], start_time=_iso(first["time"]),
                        end_time=_iso(first_ff["time"]), seconds=rehash_seconds,
                        valid=not rehash_errors, progress_count=len(lsh),
                        progress_lines=[event["line"] for event in lsh], errors=rehash_errors)
        rehash_events.append(measured)
        audit.append(measured.copy())
        warnings.extend(f"batch {attempt['batch']}, line {start['line']}: {error}" for error in rehash_errors)
    elif need_rehash:
        signal_lines = [event["line"] for event in need_rehash]
        rehash_errors.append(f"line {signal_lines[0]}: Need Rehash was reported but no LSH node 0 "
                             "start event was recorded; refresh duration is unknown")
        rehash_seconds = None
        missing = dict(identity, kind="interval", component="rehash", start_line=None,
                       end_line=first_ff["line"], start_time=None, end_time=_iso(first_ff["time"]),
                       seconds=None, valid=False, progress_count=0, progress_lines=[],
                       signal_lines=signal_lines, errors=rehash_errors)
        rehash_events.append(missing)
        audit.append(missing.copy())
        warnings.extend(f"batch {attempt['batch']}, line {start['line']}: {error}" for error in rehash_errors)
    audit.append(dict(identity, kind="interval", component="batch", start_line=start["line"],
                      end_line=end["line"], start_time=_iso(start["time"]), end_time=_iso(end["time"]),
                      seconds=seconds, valid=True))
    return dict(identity, header_line=start["line"], end_line=end["line"],
                start_time=_iso(start["time"]), end_time=_iso(end["time"]), reported_seconds=seconds,
                rehash_s=rehash_seconds, rehash_present=bool(lsh or need_rehash), rehash_valid=not rehash_errors,
                rehash_events=rehash_events, rehash_event_count=len(rehash_events), errors=rehash_errors,
                ff_count=counts["ff"], bp_count=counts["bp"], free_count=counts["free"],
                update_s=update_seconds, update_start_line=update["line"], first_ff_line=first_ff["line"])


def parse_console(text, batchsize):
    """Return batches, audit, and warnings using validated consecutive headers.

    Batch numbers are global identities.  The latest complete occurrence wins;
    an unfinished or structurally corrupt retry never replaces an older valid
    occurrence.  A usable total with damaged LSH timestamps is retained with
    ``rehash_valid=False`` and ``rehash_s=None`` so a caller can reject refresh
    adjustment without inventing a value.  No final batch is inferred complete
    from EOF, status, or an opaque ``Layer takes`` timer.

    A rehash begins at node 0 and subsequent node progress must strictly
    increase.  A missing initial progress record or a node-number reset cannot
    be interpreted as a complete refresh interval or as multiple refreshes.
    ``Need Rehash`` without LSH progress indicates an unmeasurable refresh,
    never a zero-duration refresh; that signal's time is not a start boundary.
    """
    if isinstance(batchsize, bool) or not isinstance(batchsize, int) or batchsize <= 0:
        raise ValueError("batchsize must be a positive integer")
    audit, warnings, completed = [], [], []
    occurrence = defaultdict(int)
    segment, has_headers = 0, False
    pending = None
    lines = text.splitlines()

    def discard(end_line, reason):
        if pending is not None:
            audit.append(dict(_identity(pending), kind="incomplete_batch", action="discarded",
                              header_line=pending["start"]["line"], end_line=end_line, reason=reason))
            warnings.append(f"batch {pending['batch']}, line {pending['start']['line']}: "
                            f"discarded incomplete attempt ({reason})")

    def boundary(source_line, reason):
        nonlocal segment, has_headers, pending
        discard(source_line, reason)
        pending = None
        if has_headers:
            segment += 1
            audit.append({"kind": "execution_boundary", "source_line": source_line,
                          "segment": segment, "reason": reason})
            has_headers = False

    for number, raw in enumerate(lines, 1):
        event = _line(raw, number)
        reason = _execution_boundary(event["label"])
        if reason:
            boundary(number, reason)
            continue
        header = _BATCH.fullmatch(event["label"])
        if header:
            batch = int(header[1])
            if pending is not None:
                if batch == pending["batch"] + 1:
                    row = _finish(pending, event, batchsize, audit, warnings)
                    if row is not None:
                        completed.append(row)
                else:
                    boundary(number, f"non-consecutive batch header: expected {pending['batch'] + 1}, got {batch}")
            occurrence[batch] += 1
            pending = {"batch": batch, "occurrence": occurrence[batch], "segment": segment,
                       "start": event, "events": []}
            has_headers = True
        elif pending is not None:
            pending["events"].append(event)
    discard(len(lines), "EOF has no next consecutive batch header")

    latest = {}
    for row in completed:
        batch = row["batch"]
        if batch in latest:
            prior = latest[batch]
            audit.append(dict(_identity(prior), kind="duplicate_batch", action="discarded",
                              header_line=prior["header_line"], end_line=prior["end_line"],
                              reason="superseded by latest complete occurrence",
                              kept_header_line=row["header_line"], kept_occurrence=row["occurrence"],
                              kept_segment=row["segment"]))
            warnings.append(f"batch {batch}: kept complete occurrence {row['occurrence']} "
                            f"at line {row['header_line']}; discarded occurrence {prior['occurrence']}")
        latest[batch] = row
    return {"batches": sorted(latest.values(), key=lambda row: row["header_line"]),
            "audit": audit, "warnings": warnings}
