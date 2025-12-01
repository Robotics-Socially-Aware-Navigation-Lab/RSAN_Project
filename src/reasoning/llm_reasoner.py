"""
llm_reasoner.py
----------------
Purpose:
    Provide LLM-based semantic reasoning on top of YOLO detections
    and indoor scene classification.

Features:
    - Runtime API:
        * llm_reason_from_detections(room_label, detections, ...)
          Uses indoor scene label + YOLO detections to generate
          a structured, robot-friendly description.
    - Backwards-compatible CLI:
        * llm_reason()
          Reads a JSON file from outputs/detections and writes
          reasoning_output.txt to outputs/.

Behavior:
    - Prints FULL reasoning to the terminal
    - Saves FULL reasoning to outputs/reasoning_output.txt
    - Unified pipeline builds its OWN 1-sentence summary using
      rule-based logic (no extra LLM calls).

Configuration:
    - configs/llm_config.yaml (optional)
        model: "gpt-4o-mini"
        temperature: 0.4
        prompt_template: base template used by _build_prompt()

Notes:
    - If OPENAI_API_KEY is missing, the system falls back gracefully and
      returns a short fallback string instead of crashing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.utils.file_utils import load_paths, load_yaml
from src.utils.logger import get_logger

log = get_logger("llm_reasoner")

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LLM_CONFIG_PATH = PROJECT_ROOT / "configs" / "llm_config.yaml"

try:
    _llm_cfg: Dict[str, Any] = load_yaml(LLM_CONFIG_PATH)
except FileNotFoundError:
    log.warning("llm_config.yaml not found at %s, using defaults.", LLM_CONFIG_PATH)
    _llm_cfg = {}

LLM_MODEL: str = _llm_cfg.get("model", "gpt-4o-mini")
LLM_TEMPERATURE: float = float(_llm_cfg.get("temperature", 0.4))

BASE_PROMPT_TEMPLATE: str = _llm_cfg.get(
    "prompt_template",
    """
You are an AI perception and navigation assistant for a mobile robot.

You do NOT see the raw image. Instead, you only get:
- An indoor room type classification
- A set of detected objects with labels, confidences, and bounding boxes

Room type (from indoor classifier):
{room_label}

Detected objects (from YOLO or similar detector):
{detection_summary}

Use ONLY this information.
Do NOT hallucinate objects that are not mentioned.
Use clear, robot-friendly language.

TASK:
- Describe the scene in 4–6 sentences.
- Mention important objects and what they imply about the environment.
- Identify possible navigation hazards or social constraints.
- Give a concrete navigation recommendation for the robot.
""",
)

# ---------------------------------------------------------------------
# OpenAI client (lazy singleton)
# ---------------------------------------------------------------------
_client: Optional[OpenAI] = None


def _get_client() -> Optional[OpenAI]:
    """Lazily create and cache an OpenAI client, or return None if no API key."""
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        log.warning("OPENAI_API_KEY not set; LLM reasoning will be disabled.")
        return None

    _client = OpenAI(api_key=api_key)
    return _client


# ---------------------------------------------------------------------
# Detection summarization
# ---------------------------------------------------------------------
def _summarize_detections(detections: Any) -> List[Dict[str, Any]]:
    """
    Convert various detection formats into a list of dicts:
        { "label": str, "confidence": float, "bbox": [x1, y1, x2, y2] }

    Supports:
        - ultralytics.yolo.engine.results.Results
        - list[Results]
        - list of RSAN Detection dataclasses (class_name, confidence, bbox_xyxy)
        - list of dicts with keys 'label' or 'class_name'
        - list of strings (labels only, no bbox/score)
    """
    if detections is None:
        return []

    # Already list of dicts
    if isinstance(detections, (list, tuple)) and detections and isinstance(detections[0], dict):
        out: List[Dict[str, Any]] = []
        for d in detections:
            label = d.get("label") or d.get("class_name") or "unknown"
            conf = float(d.get("confidence", 1.0))
            bbox = d.get("bbox") or d.get("bbox_xyxy") or None
            out.append({"label": str(label), "confidence": conf, "bbox": bbox})
        return out

    # List of RSAN Detection dataclasses
    if isinstance(detections, (list, tuple)) and detections and hasattr(detections[0], "class_name"):
        out = []
        for d in detections:
            out.append(
                {
                    "label": str(d.class_name),
                    "confidence": float(getattr(d, "confidence", 1.0)),
                    "bbox": list(getattr(d, "bbox_xyxy", (None, None, None, None))),
                }
            )
        return out

    # List of labels only
    if isinstance(detections, (list, tuple)) and detections and isinstance(detections[0], str):
        return [{"label": str(lbl), "confidence": 1.0, "bbox": None} for lbl in detections]

    # ultralytics Results type
    try:
        from ultralytics.yolo.engine.results import Results  # type: ignore

        if isinstance(detections, list) and detections and isinstance(detections[0], Results):
            detections = detections[0]

        if isinstance(detections, Results):
            out = []
            boxes = detections.boxes
            if boxes is None:
                return []
            xyxy = boxes.xyxy.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            names = detections.names
            for b, c, s in zip(xyxy, cls_ids, confs):
                out.append(
                    {
                        "label": str(names[int(c)]),
                        "confidence": float(s),
                        "bbox": [float(v) for v in b.tolist()],
                    }
                )
            return out
    except Exception as exc:  # pragma: no cover
        log.debug("Failed to parse detections as ultralytics Results: %s", exc)

    # Fallback: unknown format
    return []


def _extract_labels(summary: List[Dict[str, Any]]) -> List[str]:
    """Convenience: extract just the labels from summarized detections."""
    return [s["label"] for s in summary]


# ---------------------------------------------------------------------
# Prompt building & LLM call
# ---------------------------------------------------------------------
def _build_prompt(room_label: str, det_summary: List[Dict[str, Any]]) -> str:
    if det_summary:
        summary_str = json.dumps(det_summary, indent=2)
    else:
        summary_str = "[]  # no objects detected"

    return BASE_PROMPT_TEMPLATE.format(
        room_label=room_label,
        detection_summary=summary_str,
        objects=", ".join(_extract_labels(det_summary)) or "none",
    )


def _call_llm(prompt: str) -> str:
    client = _get_client()
    if client is None:
        return "LLM disabled (no OPENAI_API_KEY set)."

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:  # pragma: no cover - network/API error path
        log.warning("LLM call failed: %s", exc)
        return "LLM reasoning failed; fallback used."


# ---------------------------------------------------------------------
# PUBLIC RUNTIME API
# ---------------------------------------------------------------------
def llm_reason_from_detections(
    room_label: str,
    detections: Any,
    source: Optional[str] = None,
    save: bool = True,
    print_to_console: bool = True,
) -> str:
    """
    High-level runtime API used by the unified pipeline.

    Args:
        room_label: str
            Indoor scene label from IndoorClassifier (e.g., 'hallway', 'office').
        detections: Any
            YOLO detections (Ultralytics Results object or RSAN Detection list).
        source: Optional[str]
            String identifying the source image/video (for logs).
        save: bool
            If True, writes FULL reasoning to reasoning_output.txt.
        print_to_console: bool
            If True, prints FULL reasoning to stdout.

    Returns:
        str: FULL reasoning (multi-sentence) from the LLM.
             The unified pipeline will create a 1-sentence summary
             using rule-based logic.
    """
    det_summary = _summarize_detections(detections)
    prompt = _build_prompt(room_label, det_summary)
    full_reasoning = _call_llm(prompt)

    # Print to console
    if print_to_console:
        header = f"=== LLM Scene Reasoning (source={source or 'N/A'}) ==="
        print("\n" + header)
        print(full_reasoning)
        print("=" * len(header) + "\n")

    # Save full reasoning to file
    if save:
        paths = load_paths()
        out_path = paths["results"] / "reasoning_output.txt"
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                f.write(full_reasoning)
            log.info("LLM reasoning saved to %s", out_path)
        except Exception as exc:
            log.warning("Failed to save reasoning_output.txt: %s", exc)

    if source:
        log.info("LLM reasoning computed for source: %s", source)

    return full_reasoning


# ---------------------------------------------------------------------
# BACKWARDS-COMPATIBLE CLI MODE
# ---------------------------------------------------------------------
def llm_reason() -> None:
    """
    Legacy CLI entry point.

    Reads the first *.json file in outputs/detections/, summarizes it,
    calls the LLM with a generic room label ("unknown-room"), and writes
    outputs/reasoning_output.txt (FULL reasoning).
    """
    paths = load_paths()
    detections_dir = paths["detections"]

    json_files = sorted(detections_dir.glob("*.json"))
    if not json_files:
        log.error("No detection JSON files found in %s", detections_dir)
        return

    detection_file = json_files[0]
    log.info("Using detection file: %s", detection_file)

    with open(detection_file, "r") as f:
        detections = json.load(f)

    det_summary = _summarize_detections(detections)
    prompt = _build_prompt("unknown-room", det_summary)
    full_reasoning = _call_llm(prompt)

    out_path = paths["results"] / "reasoning_output.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(full_reasoning)

    log.info("LLM reasoning saved to %s", out_path)
    print("\n=== LLM Scene Reasoning (legacy JSON mode) ===")
    print(full_reasoning)
    print("==============================================\n")


if __name__ == "__main__":
    # CLI: python -m src.reasoning.llm_reasoner
    llm_reason()
