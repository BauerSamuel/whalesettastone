"""
Pytest configuration: use non-interactive matplotlib and close figures after
every test so plots never linger or stay open, even when a test fails.
"""

# Set Agg first, before any other code can import pyplot and trigger a GUI backend
import matplotlib
matplotlib.use("Agg")

import pytest


def _close_all_figures():
    """Close every matplotlib figure; never raise."""
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass


@pytest.fixture(autouse=True)
def close_matplotlib_figures():
    """Close all figures before and after each test so none are left open on failure."""
    _close_all_figures()
    try:
        yield
    finally:
        _close_all_figures()


def pytest_runtest_teardown(item):
    """Hook: close all figures again after each test teardown (backup if fixture is skipped)."""
    _close_all_figures()


def pytest_sessionfinish(session, exitstatus):
    """Close all figures at end of test session so nothing is left open."""
    _close_all_figures()
