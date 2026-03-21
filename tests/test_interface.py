"""
Test for the whale audio interface.

Builds only download and upload tabs (no full visualizer) so the test is fast
and does not block or open plot windows.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.audio_interface import AudioInterface


def test_interface():
    """Build download and upload tabs without the visualizer (avoids blocking/plots)."""
    iface = AudioInterface()
    download_tab = iface.create_download_tab()
    upload_tab = iface.create_upload_tab()
    assert download_tab is not None and hasattr(download_tab, "children")
    assert upload_tab is not None and hasattr(upload_tab, "children")


if __name__ == "__main__":
    test_interface()
