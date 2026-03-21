"""
Unit tests for the Streamlit state manager.

These tests exercise session-state initialization and batch bookkeeping
without needing a real Streamlit runtime.
"""

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import src.state_manager as state_manager_module
from src.state_manager import StateManager


class DummySessionState(dict):
    """Simple dict-backed stand‑in for st.session_state."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class DummyStreamlit:
    def __init__(self):
        self.session_state = DummySessionState()


@pytest.fixture
def dummy_st(monkeypatch):
    """Provide a patched Streamlit module with an in-memory session_state."""
    dummy = DummyStreamlit()
    monkeypatch.setattr(state_manager_module, "st", dummy)
    return dummy


def test_initialize_state_sets_expected_defaults(dummy_st):
    """initialize_state should create all known keys with sensible defaults."""
    StateManager.initialize_state()

    ss = dummy_st.session_state
    assert isinstance(ss.audio_files, dict)
    assert isinstance(ss.batches, dict)
    assert ss.current_batch is None
    assert ss.files_uploaded is False
    assert ss.batch_processed is False


def test_create_new_batch_and_mark_processed(dummy_st):
    """Creating and marking a batch processed should update state consistently."""
    StateManager.initialize_state()
    files = ["file1.wav", "file2.wav"]

    StateManager.create_new_batch(files)

    ss = dummy_st.session_state
    assert ss.files_uploaded is True
    assert ss.current_batch in ss.batches
    assert ss.batches[ss.current_batch]["files"] == files
    assert ss.batches[ss.current_batch]["processed"] is False

    # Before marking processed
    assert StateManager.is_batch_processed() is False

    StateManager.mark_batch_processed()

    assert ss.batches[ss.current_batch]["processed"] is True
    assert StateManager.is_batch_processed() is True


def test_get_current_batch_files_and_debug_info(dummy_st):
    """get_current_batch_files and get_debug_info should reflect internal state."""
    StateManager.initialize_state()

    # No batch yet
    assert StateManager.get_current_batch_files() == []
    debug = StateManager.get_debug_info()
    assert debug["files_uploaded"] is False
    assert debug["current_batch"] is None
    assert debug["batch_processed"] is False
    assert debug["batch_keys"] == []

    # After creating a batch
    files = ["a.wav", "b.wav"]
    StateManager.create_new_batch(files)
    current_files = StateManager.get_current_batch_files()
    assert current_files == files

    debug_after = StateManager.get_debug_info()
    assert debug_after["files_uploaded"] is True
    assert debug_after["current_batch"] is not None
    assert debug_after["batch_exists"] is True
    assert debug_after["batch_processed"] is False
    assert len(debug_after["batch_keys"]) == 1

