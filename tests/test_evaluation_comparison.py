"""
Tests for evaluation comparison system.
"""

import pytest
import json
from unittest.mock import Mock, patch

from src.stylometric.evaluation_comparison import (
    EvaluationComparator,
    ComparisonResult,
    MetricStatistics,