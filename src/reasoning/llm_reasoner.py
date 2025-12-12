"""
llm_reasoner.py (SAFE VERSION)
--------------------------------

Purpose:
    Provide optional LLM-based semantic reasoning on top of YOLO detections
    and indoor scene classification.

Safe Behavior:
    - If OPENAI_API_KEY is missing:
        ✔ No errors
        ✔ No API calls
        ✔ Returns a clean fallback message
        ✔ Pipeline continues normally

    - If the key exists:
        ✔ Full LLM reasoning works
        ✔ Output saved to outputs/reasoning_output.txt

This makes the RSAN project safe for teams.
"""

# ================================================================
# LLM Reasoner
# ================================================================

from __future__ import annotations

# ================================================================
# Standard Library Imports
# ================================================================
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# ================================================================
# Third-Party Imports
# ================================================================
from dotenv import load_dotenv
from openai import OpenAI

# ================================================================
# RSAN Utility Imports
# ================================================================
from src.utils.file_utils import load_paths, load_yaml
from src.utils.logger import get_logger

# Load environment variables safely (must be AFTER imports)
load_dotenv()

log = get_logger("llm_reasoner")


# ---------------------------------------------------------------------
# Load configuration (optional)
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LLM_CONFIG_PATH = PROJECT_ROOT / "configs" / "llm_config.yaml"

try:
    _llm_cfg: Dict[str, Any] = load_yaml(LLM_CONFIG_PATH)
except FileNotFoundError:
    log.warning("llm_config.yaml not found — using defaults.")
    _llm_cfg = {}

LLM_MODEL: str = _llm_cfg.get("model", "gpt-4o-mini")
LLM_TEMPERATURE: float = float(_llm_cfg.get("temperature", 0.4))

BASE_PROMPT_TEMPLATE: str = _llm_cfg.get(
    "prompt_template",
    """
You are an AI perception assistant.

Room type:
{room_label}

Detected objects:
{detection_summary}

Use ONLY the information provided.

TASK:
- Describe the scene in 4–6 sentences.
- Mention key objects.
- Mention navigation hazards.
- Make a navigation recommendation.
""",
)

# ---------------------------------------------------------------------
# OpenAI client (safe + optional)
# ---------------------------------------------------------------------
_client: Optional[OpenAI] = None


# def _get_client() -> Optional[OpenAI]:
#     """
#     Create the OpenAI client ONLY if the API key is available.
#     Compatible with OpenAI 1.x+ (no constructor arguments allowed).
#     """
#     global _client
#     if _client is not None:
#         return _client

#     api_key = os.getenv("OPENAI_API_KEY")

#     if not api_key:
#         log.warning("OPENAI_API_KEY is not set → LLM reasoning disabled.")
#         return None

#     try:
#         # Modern OpenAI SDK: constructor takes NO arguments.
#         # Key must exist in environment before this call.
#         _client = OpenAI()
#         return _client

#     except Exception as exc:
#         log.warning("Failed to initialize OpenAI client: %s", exc)
#         return None


def _get_client() -> Optional[OpenAI]:
    """
    Safe OpenAI client initializer.
    Explicitly passes the API key to avoid proxy injection problems.
    """
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        log.warning("OPENAI_API_KEY is not set → LLM reasoning disabled.")
        return None

    try:
        # FIX: pass api_key explicitly to avoid unexpected constructor args
        _client = OpenAI(api_key=api_key)
        return _client

    except Exception as exc:
        log.warning(f"Failed to initialize OpenAI client: {exc}")
        return None


# ---------------------------------------------------------------------
# Detection → simple Python dicts
# ---------------------------------------------------------------------
def _summarize_detections(detections: Any) -> List[Dict[str, Any]]:
    """Convert detector output into a uniform list of dicts."""
    if detections is None:
        return []

    # Already dict format
    if isinstance(detections, list) and detections and isinstance(detections[0], dict):
        out = []
        for d in detections:
            out.append(
                {
                    "label": d.get("label") or d.get("class_name", "unknown"),
                    "confidence": float(d.get("confidence", 1.0)),
                    "bbox": d.get("bbox") or d.get("bbox_xyxy"),
                }
            )
        return out

    # RSAN dataclasses
    if hasattr(detections, "__iter__") and detections and hasattr(detections[0], "class_name"):
        out = []
        for d in detections:
            out.append(
                {
                    "label": d.class_name,
                    "confidence": float(getattr(d, "confidence", 1.0)),
                    "bbox": list(getattr(d, "bbox_xyxy", [])),
                }
            )
        return out

    # List of strings
    if isinstance(detections, list) and detections and isinstance(detections[0], str):
        return [{"label": lbl, "confidence": 1.0, "bbox": None} for lbl in detections]

    # Ultralytics YOLO result
    try:
        from ultralytics.yolo.engine.results import Results

        if isinstance(detections, list) and detections and isinstance(detections[0], Results):
            detections = detections[0]

        if isinstance(detections, Results):
            boxes = detections.boxes
            if boxes is None:
                return []

            xyxy = boxes.xyxy.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            names = detections.names

            out = []
            for b, cid, s in zip(xyxy, cls_ids, confs):
                out.append(
                    {
                        "label": names[int(cid)],
                        "confidence": float(s),
                        "bbox": [float(v) for v in b.tolist()],
                    }
                )
            return out
    except Exception:
        pass

    return []  # fallback


# ---------------------------------------------------------------------
# Build LLM prompt
# ---------------------------------------------------------------------
def _build_prompt(room_label: str, det_summary: List[Dict[str, Any]]) -> str:
    return BASE_PROMPT_TEMPLATE.format(
        room_label=room_label,
        detection_summary=json.dumps(det_summary, indent=2),
    )


# ---------------------------------------------------------------------
# Safe API call
# ---------------------------------------------------------------------
def _call_llm(prompt: str) -> str:
    """
    Wrapper around OpenAI API.
    Always safe. Never crashes.
    """
    client = _get_client()
    if client is None:
        return "LLM disabled (no API key)."

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()

    except Exception as exc:
        log.warning("LLM call failed: %s", exc)
        return "LLM reasoning failed; fallback used."


# ---------------------------------------------------------------------
# PUBLIC API (used by run_unified_pipeline)
# ---------------------------------------------------------------------
def llm_reason_from_detections(
    room_label: str,
    detections: Any,
    source: Optional[str] = None,
    save: bool = True,
    print_to_console: bool = True,
) -> str:

    det_summary = _summarize_detections(detections)
    prompt = _build_prompt(room_label, det_summary)
    full_text = _call_llm(prompt)

    if print_to_console:
        print("\n=== LLM Scene Reasoning ===")
        print(full_text)
        print("===========================\n")

    # Save output
    if save:
        paths = load_paths()
        out_file = paths["results"] / "reasoning_output.txt"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w") as f:
            f.write(full_text)
        log.info("Saved full reasoning to %s", out_file)

    return full_text


# ---------------------------------------------------------------------
# Legacy CLI entry point
# ---------------------------------------------------------------------
def llm_reason():
    paths = load_paths()
    json_files = sorted(paths["detections"].glob("*.json"))
    if not json_files:
        print("No detection JSON files found.")
        return

    with open(json_files[0], "r") as f:
        detections = json.load(f)

    prompt = _build_prompt("unknown-room", _summarize_detections(detections))
    print(_call_llm(prompt))


if __name__ == "__main__":
    llm_reason()
