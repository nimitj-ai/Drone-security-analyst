"""
vlm_captioner.py
----------------
Vision-Language Model pipeline for per-frame semantic captioning and
object detection.

Supported backends (set VLM_BACKEND env var or pass to constructor):
  - "blip2"     : Salesforce BLIP-2 (local GPU/CPU, recommended)
  - "florence2" : Microsoft Florence-2 (fast, local)
  - "mock"      : Rule-based captioner from EXIF/filename tags (no GPU needed)

The captioner returns a FrameCaption dataclass with:
  - caption     : str  — a natural-language description of the frame
  - objects     : List[str] — detected entity labels
  - confidence  : float — model confidence (where available)
"""

import os
import re
from dataclasses import dataclass, field
from typing import List
from PIL import Image


@dataclass
class FrameCaption:
    frame_id: int
    caption: str
    objects: List[str] = field(default_factory=list)
    confidence: float = 1.0
    raw_output: str = ""


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------
class BaseCaptioner:
    def caption(self, frame_id: int, image: Image.Image) -> FrameCaption:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Mock captioner (no GPU required — useful for CI and demos)
# ---------------------------------------------------------------------------
_MOCK_CAPTIONS = [
    ("A person in dark clothing standing near the perimeter chain-link fence at night.",
     ["Person (dark clothing)"]),
    ("Two individuals loitering at the main gate. One is carrying a backpack.",
     ["Person", "Person", "Backpack"]),
    ("A white pickup truck parked outside the restricted perimeter fence.",
     ["White Pickup Truck"]),
    ("Empty patrol zone. No suspicious activity detected.",
     []),
    ("A blue sedan driving slowly along the east fence line.",
     ["Blue Sedan"]),
    ("A security guard performing a routine check near the loading bay.",
     ["Security Guard"]),
    ("An unidentified person climbing over the perimeter fence.",
     ["Person"]),
    ("A forklift operating inside the loading bay. No anomalies detected.",
     ["Forklift"]),
    ("Abandoned backpack spotted near the south gate with no person in vicinity.",
     ["Backpack"]),
    ("Daytime patrol. Clear skies. No people or vehicles detected.",
     []),
]


class MockCaptioner(BaseCaptioner):
    """Deterministic round-robin captions — zero ML dependencies."""

    def __init__(self):
        self._idx = 0

    def caption(self, frame_id: int, image: Image.Image) -> FrameCaption:
        cap_text, objs = _MOCK_CAPTIONS[self._idx % len(_MOCK_CAPTIONS)]
        self._idx += 1
        return FrameCaption(
            frame_id=frame_id,
            caption=cap_text,
            objects=objs,
            confidence=0.95,
            raw_output=cap_text,
        )


# ---------------------------------------------------------------------------
# BLIP-2 captioner (requires transformers + torch + GPU recommended)
# ---------------------------------------------------------------------------
class Blip2Captioner(BaseCaptioner):
    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b"):
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[BLIP-2] Loading {model_name} on {self.device} …")
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.model.eval()

    def caption(self, frame_id: int, image: Image.Image) -> FrameCaption:
        import torch
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=80)
        cap_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        objects = _extract_objects_from_caption(cap_text)
        return FrameCaption(
            frame_id=frame_id,
            caption=cap_text,
            objects=objects,
            confidence=0.9,
            raw_output=cap_text,
        )


# ---------------------------------------------------------------------------
# Florence-2 captioner (smaller, faster than BLIP-2)
# ---------------------------------------------------------------------------
class Florence2Captioner(BaseCaptioner):
    def __init__(self, model_name: str = "microsoft/Florence-2-base"):
        from transformers import AutoProcessor, AutoModelForCausalLM
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Florence-2] Loading {model_name} on {self.device} …")
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    def caption(self, frame_id: int, image: Image.Image) -> FrameCaption:
        import torch
        task_prompt = "<CAPTION>"
        inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=128,
                num_beams=3,
            )
        output = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        cap_text = self.processor.post_process_generation(
            output, task=task_prompt, image_size=(image.width, image.height)
        ).get("<CAPTION>", output).strip()
        objects = _extract_objects_from_caption(cap_text)
        return FrameCaption(
            frame_id=frame_id,
            caption=cap_text,
            objects=objects,
            confidence=0.9,
            raw_output=cap_text,
        )


# ---------------------------------------------------------------------------
# Utility: extract entity keywords from free-text captions
# ---------------------------------------------------------------------------
_ENTITY_KEYWORDS = {
    "person": ["person", "individual", "man", "woman", "pedestrian", "intruder"],
    "vehicle": ["car", "sedan", "truck", "pickup", "van", "vehicle", "suv", "forklift"],
    "backpack": ["backpack", "bag", "package", "suitcase", "box"],
    "guard": ["guard", "officer", "security"],
    "fence": ["fence", "perimeter", "gate", "wall"],
    "climbing": ["climbing", "jumping", "crawling", "scaling"],
}


def _extract_objects_from_caption(caption: str) -> List[str]:
    found = []
    lower = caption.lower()
    for label, keywords in _ENTITY_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            found.append(label)
    return found


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_captioner(backend: str | None = None) -> BaseCaptioner:
    """
    Returns the appropriate captioner.
    Priority: argument > VLM_BACKEND env var > "mock"
    """
    backend = backend or os.getenv("VLM_BACKEND", "mock")
    backend = backend.lower()
    if backend == "blip2":
        return Blip2Captioner()
    elif backend == "florence2":
        return Florence2Captioner()
    elif backend == "mock":
        return MockCaptioner()
    else:
        raise ValueError(f"Unknown VLM backend: {backend!r}. Choose blip2, florence2, or mock.")
