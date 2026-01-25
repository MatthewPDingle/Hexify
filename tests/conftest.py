"""Pytest configuration and shared fixtures for Hexify tests."""
import sys
import os
import pytest
import numpy as np
import cv2

# Add parent directory to path to import hexify modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Paths
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_DIR = os.path.join(TESTS_DIR, 'fixtures')
REFERENCE_DIR = os.path.join(TESTS_DIR, 'reference_outputs')


@pytest.fixture
def fixtures_dir():
    """Return the path to the fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def reference_dir():
    """Return the path to the reference outputs directory."""
    return REFERENCE_DIR


@pytest.fixture
def gradient_64x64():
    """Load the 64x64 gradient fixture image."""
    path = os.path.join(FIXTURES_DIR, 'gradient_64x64.png')
    img = cv2.imread(path)
    if img is None:
        pytest.skip(f"Fixture not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@pytest.fixture
def color_blocks_64x64():
    """Load the 64x64 color blocks fixture image."""
    path = os.path.join(FIXTURES_DIR, 'color_blocks_64x64.png')
    img = cv2.imread(path)
    if img is None:
        pytest.skip(f"Fixture not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@pytest.fixture
def gradient_32x48():
    """Load the 32x48 gradient fixture image."""
    path = os.path.join(FIXTURES_DIR, 'gradient_32x48.png')
    img = cv2.imread(path)
    if img is None:
        pytest.skip(f"Fixture not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@pytest.fixture
def fire_64x64():
    """Load the 64x64 fire fixture image."""
    path = os.path.join(FIXTURES_DIR, 'fire_64x64.png')
    img = cv2.imread(path)
    if img is None:
        pytest.skip(f"Fixture not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
