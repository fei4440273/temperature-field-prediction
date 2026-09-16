from __future__ import annotations

import pytest

from sic_cu.compat import strict_zip


def test_strict_zip_pairs_equal_length_iterables() -> None:
    assert list(strict_zip([1, 2], ["a", "b"])) == [(1, "a"), (2, "b")]


def test_strict_zip_rejects_different_length_iterables() -> None:
    with pytest.raises(ValueError, match="different lengths"):
        list(strict_zip([1, 2], ["a"]))
