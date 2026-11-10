"""Smoke test: the package imports and every stage module exists."""

import importlib

STAGES = ["ingest", "pose", "transforms", "quality", "render", "pipeline"]


def test_package_imports():
    import boulder

    assert boulder.__version__


def test_stage_modules_import():
    for stage in STAGES:
        importlib.import_module(f"boulder.{stage}")
