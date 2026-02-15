"""Processing pipeline for Cadabrio.

Manages the sequence of transformations applied to assets,
from ingestion through AI processing to final export.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class PipelineStage(Enum):
    """Stages in the asset processing pipeline."""

    INGEST = "ingest"
    PREPROCESS = "preprocess"
    AI_GENERATE = "ai_generate"
    RECONSTRUCT = "reconstruct"  # photogrammetry
    REFINE = "refine"
    EXPORT = "export"


@dataclass
class PipelineStep:
    """A single step in the processing pipeline."""

    stage: PipelineStage
    operation: str
    params: dict[str, Any]
    completed: bool = False
    result: Any = None


class Pipeline:
    """Manages ordered processing steps for asset creation."""

    def __init__(self):
        self._steps: list[PipelineStep] = []
        self._current_index: int = 0

    def add_step(self, stage: PipelineStage, operation: str, **params):
        """Add a processing step to the pipeline."""
        self._steps.append(PipelineStep(stage=stage, operation=operation, params=params))

    def execute_next(self) -> PipelineStep | None:
        """Execute the next pending step. Returns the step or None if done."""
        if self._current_index >= len(self._steps):
            return None

        step = self._steps[self._current_index]
        # TODO: Route to appropriate handler based on stage/operation
        step.completed = True
        self._current_index += 1
        return step

    @property
    def progress(self) -> float:
        """Return pipeline progress as 0.0 to 1.0."""
        if not self._steps:
            return 1.0
        return self._current_index / len(self._steps)
