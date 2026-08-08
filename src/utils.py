"""Shared helper utilities for the training and serving pipelines."""

import os


def get_project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_data_path(filename: str = "wine.csv") -> str:
    return os.path.join(get_project_root(), "data", filename)


def get_models_dir() -> str:
    return os.path.join(get_project_root(), "models")
