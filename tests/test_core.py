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

    def test_save_and_load(self, tmp_path):
        project = Project.new("TestProject", "print")
        project.add_asset({"type": "image", "path": "test.png", "width": 100, "height": 100})
        assert project.dirty is True

        save_path = tmp_path / "test.cadabrio"
        project.save(save_path)
        assert save_path.exists()
        assert project.dirty is False

        loaded = Project.load(save_path)
        assert loaded.name == "TestProject"
        assert loaded.target == ProjectTarget.PRINT
        assert len(loaded.assets) == 1
        assert loaded.assets[0]["type"] == "image"
        assert loaded.dirty is False

    def test_save_appends_extension(self, tmp_path):
        project = Project.new("Foo")
        project.save(tmp_path / "Foo")
        assert (tmp_path / "Foo.cadabrio").exists()

    def test_new_project(self):
        project = Project.new("Fresh", "blender")
        assert project.name == "Fresh"
        assert project.target == ProjectTarget.BLENDER
        assert project.assets == []
        assert project.dirty is False


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
