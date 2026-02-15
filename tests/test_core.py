"""Tests for core modules."""

from cadabrio.core.scale_manager import ScaleManager, Unit
from cadabrio.core.project import Project, ProjectTarget
from cadabrio.core.pipeline import Pipeline, PipelineStage


class TestScaleManager:
    def test_mm_to_meters(self):
        sm = ScaleManager()
        result = sm.convert(1000, Unit.MILLIMETERS, Unit.METERS)
        assert abs(result - 1.0) < 1e-9

    def test_inches_to_mm(self):
        sm = ScaleManager()
        result = sm.convert(1.0, Unit.INCHES, Unit.MILLIMETERS)
        assert abs(result - 25.4) < 1e-6

    def test_round_trip_conversion(self):
        sm = ScaleManager()
        original = 42.5
        meters = sm.convert(original, Unit.CENTIMETERS, Unit.METERS)
        back = sm.convert(meters, Unit.METERS, Unit.CENTIMETERS)
        assert abs(back - original) < 1e-9


class TestProject:
    def test_default_project(self):
        project = Project()
        assert project.name == "Untitled"
        assert project.target == ProjectTarget.GENERAL

    def test_project_target(self):
        project = Project(target=ProjectTarget.PRINT)
        assert project.target == ProjectTarget.PRINT


class TestPipeline:
    def test_empty_pipeline_progress(self):
        pipe = Pipeline()
        assert pipe.progress == 1.0

    def test_pipeline_execution(self):
        pipe = Pipeline()
        pipe.add_step(PipelineStage.INGEST, "import_image", path="test.png")
        pipe.add_step(PipelineStage.AI_GENERATE, "text_to_3d", prompt="a cube")
        assert pipe.progress == 0.0

        step = pipe.execute_next()
        assert step is not None
        assert step.completed
        assert pipe.progress == 0.5

        step = pipe.execute_next()
        assert step.completed
        assert pipe.progress == 1.0

        step = pipe.execute_next()
        assert step is None
