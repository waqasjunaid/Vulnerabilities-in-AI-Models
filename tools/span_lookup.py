#!/usr/bin/env python3
"""Turn frozen sentence scores into the spans guardian v2 already consumes.

Marker spans are unchanged: a field that contains an instruction marker still
contributes the text from the marker to the end of the field. A sentence whose
detector score reaches the threshold is added as its own span, unless that
sentence is already inside a marker span.

An observation with neither a marker nor a flagged sentence contributes no
span. The whole observation is not used as a span. Every span returned here is
located, so the second value is always False.
"""
from __future__ import annotations


def sentence_spans(text, scores, threshold, split, marker_re, content_fields):
    """Return (spans, unexplained, sentences_checked, sentences_missing)."""
    spans = []
    for field in content_fields(text or ""):
        m = marker_re.search(field)
        if m:
            spans.append(field[m.start():])
    checked = missing = 0
    for sent in split(text or ""):
        checked += 1
        score = scores.get(sent)
        if score is None:
            missing += 1
            continue
        if score >= threshold and not any(sent in span for span in spans):
            spans.append(sent)
    return spans, False, checked, missing
