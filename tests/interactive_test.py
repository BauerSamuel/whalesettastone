"""
Test for the interactive whale audio interface.

In pytest we only verify AudioInterface can be constructed and that the tab
structure is buildable with minimal features (no real file I/O or plotting).
For full UI in Jupyter, use launch_interactive_interface().
"""

import sys
from pathlib import Path
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.audio_interface import AudioInterface


def test_interactive():
    """Smoke-test interface: build download/upload tabs only (no visualizer to avoid blocking plots)."""
    interface = AudioInterface()
    download_tab = interface.create_download_tab()
    upload_tab = interface.create_upload_tab()
    assert download_tab is not None and hasattr(download_tab, "children")
    assert upload_tab is not None and hasattr(upload_tab, "children")


def launch_interactive_interface():
    """Call this from Jupyter to show the full interface with display()."""
    from IPython.display import display
    interface = AudioInterface()
    interface.display_welcome_message()
    display(interface.create_interface())


if __name__ == "__main__":
    test_interactive()
    print("Interface built successfully. In Jupyter, use launch_interactive_interface() to show the UI.")
