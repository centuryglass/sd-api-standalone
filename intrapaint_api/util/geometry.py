"""A minimal, Qt-free stand-in for the subset of QSize the standalone API relies on.

The original code used PySide6's QSize purely as a (width, height) value type for image dimensions and size-typed
config values. This class mirrors the small slice of QSize's API that code depends on (the `.width()` / `.height()`
accessor methods, copy construction, and equality) so call sites need no changes beyond swapping the type name.
"""
from typing import Union


class Size:
    """An immutable-ish (width, height) pair mirroring the QSize methods used by the API."""

    def __init__(self, width: Union[int, 'Size'] = 0, height: int = 0) -> None:
        if isinstance(width, Size):  # copy constructor, matching QSize(other)
            self._width = width._width
            self._height = width._height
        else:
            self._width = int(width)
            self._height = int(height)

    def width(self) -> int:
        """Return the width component."""
        return self._width

    def height(self) -> int:
        """Return the height component."""
        return self._height

    def setWidth(self, width: int) -> None:  # noqa: N802 - matches the QSize method name
        """Set the width component."""
        self._width = int(width)

    def setHeight(self, height: int) -> None:  # noqa: N802 - matches the QSize method name
        """Set the height component."""
        self._height = int(height)

    def isEmpty(self) -> bool:  # noqa: N802 - matches the QSize method name
        """Return whether either dimension is non-positive."""
        return self._width <= 0 or self._height <= 0

    def isNull(self) -> bool:  # noqa: N802 - matches the QSize method name
        """Return whether both dimensions are zero."""
        return self._width == 0 and self._height == 0

    def isValid(self) -> bool:  # noqa: N802 - matches the QSize method name
        """Return whether both dimensions are non-negative."""
        return self._width >= 0 and self._height >= 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Size):
            return NotImplemented
        return self._width == other._width and self._height == other._height

    def __hash__(self) -> int:
        return hash((self._width, self._height))

    def __repr__(self) -> str:
        return f'Size({self._width}, {self._height})'
