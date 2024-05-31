"""Maze representation and utils

This module contains a class for representing mazes and tools for loading and
saving mazes to files.
"""

from __future__ import annotations

import math
import os
import re

from dataclasses import dataclass
from enum import Flag
from functools import reduce
from operator import or_
from typing import cast, TYPE_CHECKING

import numpy as np

from .directions import Direction
from .unionfind import UnionFind

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Set
    from typing import BinaryIO, Callable, Literal, Self, TextIO, TypeAlias

    import numpy.typing as npt

    NumpyCheat = np.generic
else:
    NumpyCheat = object

type MazeSize = tuple[int, int]

AnyPath: TypeAlias = os.PathLike | str | bytes


class Walls(Flag):
    """Bit masks for the walls."""
    NORTH = 0x1
    EAST = 0x2
    SOUTH = 0x4
    WEST = 0x8

    def __bytes__(self) -> bytes:
        return self.value.to_bytes()

    def to_bytes(self) -> bytes:
        """Convert the wall specification to bytes."""
        return bytes(self)

    @classmethod
    def from_bytes(cls, byte: bytes) -> Self:
        """Create a wall specification from a byte."""
        if len(byte) != 1:
            raise ValueError(f"expected a single byte (got {len(byte)} bytes)")
        return cls(int.from_bytes(byte) & 0xF)

    @classmethod
    def none(cls) -> Self:
        """Return a wall specification for no walls."""
        return cls(0)

    @classmethod
    def all(cls) -> Self:
        """Return a wall specification for all possible walls."""
        return ~cls.none()


class LineDirection(Flag):
    """Direction for the lines in the borders of a maze."""
    # Base directions
    UP = 0x1
    RIGHT = 0x2
    DOWN = 0x4
    LEFT = 0x8

    # Lines
    VERTICAL = UP | DOWN
    HORIZONTAL = LEFT | RIGHT

    # Corners
    TOP_LEFT = DOWN | RIGHT
    TOP_RIGHT = DOWN | LEFT
    BOTTOM_LEFT = UP | RIGHT
    BOTTOM_RIGHT = UP | LEFT

    # Inner Edges
    TOP_MID = HORIZONTAL | DOWN
    BOTTOM_MID = HORIZONTAL | UP
    MID_LEFT = VERTICAL | RIGHT
    MID_RIGHT = VERTICAL | LEFT

    # Inner / Empty
    FULL = HORIZONTAL | VERTICAL
    EMPTY = 0


Charset: TypeAlias = "Callable[[LineDirection], str]"


def _ascii_box(line: LineDirection) -> str:
    match line:
        case LineDirection.EMPTY:
            return ' '
        case LineDirection.HORIZONTAL | LineDirection.LEFT | LineDirection.RIGHT:
            return '-'
        case LineDirection.VERTICAL | LineDirection.UP | LineDirection.DOWN:
            return '|'
        case _:
            return '+'


def ascii_charset(line: LineDirection) -> str:
    """A simple charset that uses ASCII characters ('-', '|', '+') to draw tables"""
    match line:
        case LineDirection.EMPTY:
            return ' '
        case LineDirection.HORIZONTAL:
            return '-'
        case LineDirection.VERTICAL:
            return '|'
        case _:
            return '+'


def utf8_charset(line: LineDirection) -> str:  # pylint: disable=too-many-return-statements
    """A simple charset that uses the UTF-8 single-line box characters to draw tables"""
    match line:
        case LineDirection.EMPTY: return ' '
        case LineDirection.HORIZONTAL: return '\u2500'
        case LineDirection.VERTICAL: return '\u2502'
        case LineDirection.TOP_LEFT: return '\u250C'
        case LineDirection.TOP_RIGHT: return '\u2510'
        case LineDirection.BOTTOM_LEFT: return '\u2514'
        case LineDirection.BOTTOM_RIGHT: return '\u2518'
        case LineDirection.TOP_MID: return '\u252C'
        case LineDirection.BOTTOM_MID: return '\u2534'
        case LineDirection.MID_LEFT: return '\u251C'
        case LineDirection.MID_RIGHT: return '\u2524'
        case LineDirection.FULL: return '\u253C'
        case LineDirection.LEFT: return '\u2574'
        case LineDirection.UP: return '\u2575'
        case LineDirection.RIGHT: return '\u2576'
        case LineDirection.DOWN: return '\u2577'
    raise AssertionError(f"impossible direction: {line!r}")


class _Missing:  # pylint: disable=too-few-public-methods
    """Indicates missing values for the maze getter."""
    __INSTANCE = None

    def __new__(cls) -> Self:
        if not cls.__INSTANCE:
            cls.__INSTANCE = super().__new__(cls)
        return cls.__INSTANCE


class Maze:
    """A class representing a maze."""

    def __init__(self, _height: int, _width: int, *, _cells: bytearray, _validate: bool = True) -> None:
        """Raw initialization of the maze, use ``Maze.empty(...)``, ``Maze.full(...)`` or ``Maze.from_*(...)``."""
        self._height = _height
        self._width = _width
        self._cells = _cells
        if _validate:
            assert len(self._cells) == self._height * self._width
            self._validate()

    @classmethod
    def empty(cls, height: int, width: int) -> Self:
        """Initializes an empty maze with the provided size."""
        size = height * width
        if not size:
            raise ValueError(f"invalid dimensions ({height}, {width}), 0 isn't allowed")

        self = cls(height, width, _cells=bytearray(Walls.none().to_bytes() * size), _validate=False)

        # Create Edges - Top Row
        for cell in range(width):
            self._cells[cell] |= Walls.NORTH.value
        # Create Edges - Rightmost Column
        for cell in range(width - 1, size, width):
            self._cells[cell] |= Walls.EAST.value
        # Create Edges - Bottom Row
        for cell in range(size - width, size):  # size - width === (height - 1) * width
            self._cells[cell] |= Walls.SOUTH.value
        # Create Edges - Leftmost Column
        for cell in range(0, size, width):
            self._cells[cell] |= Walls.WEST.value

        return self

    @classmethod
    def full(cls, height: int, width: int) -> Self:
        """Create a new full maze with the provided size."""
        size = height * width
        if not size:
            raise ValueError(f"invalid dimensions ({height}, {width}), 0 isn't allowed")

        return cls(height, width, _cells=bytearray(Walls.all().to_bytes() * size), _validate=False)

    @classmethod
    def full_from_maze(cls, maze: Maze) -> Self:
        """Create a new full maze from the provided maze."""
        return cls(maze._height, maze._width, _cells=bytearray(maze._cells), _validate=False)  # pylint: disable=protected-access

    @classmethod
    def from_maz_file(cls, maz: AnyPath | BinaryIO, size: MazeSize | None = None) -> Self:
        """
        Load a maze from a .maz file.

        Format:
            A binary file where the byte at ``row * width + col`` represents the value of the cell at ``(row, col)``.
            The value of a cell is a byte where the top nibble is 0 and the lower nibble represents the walls (see ``maze.Walls``).
            TODO: add links to the sample repos.

        size (height, width):
            If provided, use the provided dimensions.
            If not provided, the file size must either be a square size (will be detected from the content length)
            or specified in the file name in a "name.{height}x{width}.maz" format.
        """
        if isinstance(maz, AnyPath):
            maz = open(maz, "rb")
        elif isinstance(maz, bytearray | memoryview):
            raise TypeError(f"expected str, bytes, path-like or file-like. not {type(maz).__name__}")

        with maz:
            data = maz.read()

        if not size and (filename := getattr(maz, 'name', None)):
            if m := re.fullmatch(r'(?:.*\.)?(?P<height>[^0]\d*)x(?P<width>[^0]\d*)\.maz', os.path.basename(filename)):
                size = int(m['height']), int(m['width'])

        return cls.from_maz(data, size)

    @classmethod
    def from_maz(cls, data: bytes, size: MazeSize | None = None) -> Self:
        """
        Load a maze from a .maz file or file-like object.

        Format:
            A binary file where the byte at ``row * width + col`` represents the value of the cell at ``(row, col)``.
            The value of a cell is a byte where the top nibble is 0 and the lower nibble represents the walls (see ``maze.Walls``).
            TODO: add links to the sample repos.

        size (height, width):
            If provided, use the provided dimensions.
            If not provided, the file size must either be a square size (will be detected from the content length)
            or specified in the file name in a "name.{height}x{width}.maz" format.
        """
        if size:
            height, width = size
        elif (dim := math.sqrt(len(data))).is_integer():
            height = width = int(dim)
        else:
            raise ValueError("cannot detect size: not in filename and content is not a perfect square")

        return cls(height, width, _cells=bytearray(data))

    @classmethod
    def from_num_file(cls, num_file: AnyPath | TextIO, size: MazeSize | None = None) -> Self:
        """
        Load a maze from a .num file or file-like object.

        Format:
            For compatibility with the original, in this format ``(0, 0)`` is the bottom left corner
            (rather than top left) unlike the Maze class's behavior.
            The index ``(0, 0)`` is translated to ``(height - 1, 0)``.
            TODO: add format explanation and links

        size (height, width):
            If provided, use the provided dimensions.
            If not provided, the size in inferred from the file.
            When inferring, all cells must be provided.
        """
        assert list(Walls) == [Walls.NORTH, Walls.EAST, Walls.SOUTH, Walls.WEST]

        if size:
            height, width = size
            cells: list[list[Walls | None]] = [[Walls.none()] * width for _ in range(height)]
        else:
            cells = []

        if isinstance(num_file, AnyPath):
            num_file = open(num_file, "rt", encoding="ASCII")
        elif isinstance(num_file, bytearray | memoryview):
            raise TypeError(f"expected str, bytes, path-like or file-like. not {type(num_file).__name__}")

        with num_file:
            for line in num_file:
                x, y, *walls = map(int, line.split())

                if not size:
                    if len(cells) <= y:
                        cells.extend([] for _ in range(y - len(cells) + 1))
                    if len(cells[y]) <= x:
                        cells[y].extend(None for _ in range(x - len(cells[y]) + 1))

                cells[y][x] = reduce(
                    or_,
                    (wall for coord, wall in zip(walls, Walls) if coord == 1),
                    Walls.none(),
                )

        if not size:
            height = len(cells)
            if height == 0:
                raise ValueError("empty maze")
            width = len(cells[0])
            if not all(len(row) == width for row in cells):
                raise ValueError("not a square maze, must specify all cells when not specifying size")
            none_cells = [
                (row, col)
                for row, cell_row in enumerate(cells)
                for col, cell in enumerate(cell_row)
                if cell is None
            ]
            if none_cells:
                raise ValueError(f"all cells must be specified (missing: {', '.join(map(repr, none_cells))})")

        data = b''.join(
            cell.to_bytes()
            for row in cast(list[list[Walls]], cells)[::-1]
            for cell in row
        )
        return cls(height, width, _cells=bytearray(data))

    @classmethod
    def from_maze_file(cls, maze_file: AnyPath | TextIO, cell_height: int = 1, cell_width: int = 3, empty: str = ' ') -> Self:
        """
        Load a maze from a file or file-like object.

        Format:
            A text drawing of the maze.
            TODO: add format explanation and links
        """
        if isinstance(maze_file, AnyPath):
            maze_file = open(maze_file, "rt", encoding="UTF-8")
        elif isinstance(maze_file, bytearray | memoryview):
            raise TypeError(f"expected str, bytes, path-like or file-like. not {type(maze_file).__name__}")

        with maze_file:
            maze = maze_file.read()

        return cls.from_maze_text(maze, cell_height, cell_width, empty)

    @classmethod
    def from_maze_text(cls, maze: str, cell_height: int = 1, cell_width: int = 3, empty: str = ' ') -> Self:
        """
        Load a maze from a UTF-8 drawing.
        """
        lines = maze.splitlines(keepends=False)
        if not lines:
            raise ValueError("maze is empty")
        if not all(len(line) == len(lines[0]) for line in lines):
            raise ValueError("maze is not a rectangle")

        def _exact_div(a: int, b: int) -> int:
            div, mod = divmod(a, b)
            if mod:
                raise ValueError(f"invalid sizes: expected {a} can be divided by {b}")
            return div

        height = _exact_div(len(lines) - 1, cell_height + 1)
        width = _exact_div(len(lines[0]) - 1, cell_width + 1)

        def _calc_cell(row: int, col: int) -> Walls:
            cell = Walls.none()
            top_row = row * (cell_height + 1)
            bot_row = top_row + cell_height + 1
            mid_row = (top_row + bot_row) // 2
            left_col = col * (cell_width + 1)
            right_col = left_col + cell_width + 1
            mid_col = (left_col + right_col) // 2
            if lines[top_row][mid_col] != empty:
                cell |= Walls.NORTH
            if lines[mid_row][right_col] != empty:
                cell |= Walls.EAST
            if lines[bot_row][mid_col] != empty:
                cell |= Walls.SOUTH
            if lines[mid_row][left_col] != empty:
                cell |= Walls.WEST
            return cell

        data = bytearray(b''.join(
            _calc_cell(row, col).to_bytes()
            for row in range(height)
            for col in range(width)
        ))

        return cls(height, width, _cells=data)

    __NUMBERS_TO_WALLS = {
        1: Walls.WEST,
        2: Walls.NORTH,
        3: Walls.EAST,
        4: Walls.SOUTH,
        5: Walls.SOUTH | Walls.WEST,
        6: Walls.SOUTH | Walls.EAST,
        7: Walls.NORTH | Walls.EAST,
        8: Walls.NORTH | Walls.WEST,
        9: Walls.WEST | Walls.EAST,
        10: Walls.NORTH | Walls.SOUTH,
        11: Walls.SOUTH | Walls.WEST | Walls.EAST,
        12: Walls.NORTH | Walls.SOUTH | Walls.EAST,
        13: Walls.NORTH | Walls.WEST | Walls.EAST,
        14: Walls.NORTH | Walls.SOUTH | Walls.WEST,
        15: Walls.none(),
        16: Walls.all(),
    }

    @classmethod
    def from_csv_file(cls, csv_file: AnyPath | TextIO) -> Self:
        """
        Load a maze from a csv file or file-like object.

        Format:
            A csv where each row has the cells for the corresponding row
            TODO: add format explanation and links
        """
        if isinstance(csv_file, AnyPath):
            csv_file = open(csv_file, "rt", encoding="ASCII")
        elif isinstance(csv_file, bytearray | memoryview):
            raise TypeError(f"expected str, bytes, path-like or file-like. not {type(csv_file).__name__}")

        with csv_file:
            csv = csv_file.read()

        return cls.from_csv(csv)

    @classmethod
    def from_csv(cls, csv: str) -> Self:
        """
        Load a maze from a csv file or file-like object.

        Format:
            A csv where each row has the cells for the corresponding row
            TODO: add format explanation and links
        """
        lines = [
            [int(val) for val in line.split(',')]
            for line in csv.splitlines(keepends=False)
        ]
        if not lines:
            raise ValueError("maze is empty")
        if not all(len(line) == len(lines[0]) for line in lines):
            raise ValueError("maze is not a rectangle")

        height = len(lines)
        width = len(lines[0])

        data = bytearray(b''.join(
            cls.__NUMBERS_TO_WALLS[cell].to_bytes()
            for line in lines
            for cell in line
        ))

        return cls(height, width, _cells=data)

    @classmethod
    def from_file(cls, maze_file: AnyPath, fmt: Literal['maz', 'num', 'csv', 'maze', None] = None) -> Self:
        """
        Load a maze from a file or file-like object.
        If format is ``None``, it is detected from the extension:
            .maz: maz file (see ``cls.from_maz_file()``).
            .num: num file (see ``cls.from_maz_file()``).
            .csv: csv file (see ``cls.from_maz_file()``).
            default: maze file (see ``cls.from_maze_file()``).
        """
        if fmt is None:
            _, ext = os.path.splitext(maze_file)
            match ext:
                case 'maz' | b'maz': fmt = 'maz'
                case 'num' | b'num': fmt = 'num'
                case 'csv' | b'csv': fmt = 'csv'
                case 'maze' | b'maze': fmt = 'maze'
                case _:
                    # Didn't recognize extension
                    fmt = 'maze'

        match fmt:
            case 'maze': return cls.from_maze_file(maze_file)
            case 'maz': return cls.from_maz_file(maze_file)
            case 'csv': return cls.from_csv_file(maze_file)
            case 'num': return cls.from_num_file(maze_file)

    @property
    def size(self) -> MazeSize:
        """The size of the maze (height, width)."""
        return (self._height, self._width)

    @property
    def height(self) -> int:
        """The height of the maze."""
        return self._height

    @property
    def width(self) -> int:
        """The width of the maze."""
        return self._width

    @property
    def cell_size(self) -> int:
        """The amount of cells in the maze."""
        return len(self._cells)

    def _index(self, row: int, col: int) -> int:
        return row * self.width + col

    def get[T](self, row: int, col: int, default: T = None) -> Walls | T:
        """Get a cell from the maze, or ``default`` if it doesn't exist."""
        if row >= self.height or col >= self.width:
            return default
        cell = self._index(row, col)
        assert cell < self.cell_size, f"bad cell calculation for {self.size=}, ({row=}, {col=}), {cell=}"
        # if cell >= self.cell_size:
        #     return default
        return Walls(self._cells[cell])

    def __getitem__(self, idx: tuple[int, int]) -> Walls:
        if not isinstance(idx, tuple) or len(idx) != 2 or not all(isinstance(i, int) for i in idx):
            raise TypeError(f"expected 2 ints: (row, col), got {idx!r}")
        row, col = idx
        val = self.get(row, col, _Missing())
        if isinstance(val, _Missing):
            raise IndexError("TODO: pretty string")
        return Walls(val)

    def __setitem__(self, idx: tuple[int, int], value: Walls) -> None:
        if not isinstance(idx, tuple) or len(idx) != 2 or not all(isinstance(i, int) for i in idx):
            raise TypeError(f"expected 2 ints: (row, col), got {idx!r}")
        row, col = idx
        # TODO: make this prettier
        if row >= self.height:
            raise IndexError("TODO: pretty string")
        if col >= self.width:
            raise IndexError("TODO: pretty string")

        if row == 0 and not value & Walls.NORTH:
            raise ValueError("Missing NORTH wall in top row")
        if col == 0 and not value & Walls.WEST:
            raise ValueError("Missing WEST wall in leftmost column")
        if row == self.height - 1 and not value & Walls.SOUTH:
            raise ValueError("Missing SOUTH wall in bottom row")
        if col == self.width - 1 and not value & Walls.EAST:
            raise ValueError("Missing EAST wall in rightmost column")

        self._cells[self._index(row, col)] = value.value
        self._add_neighbor_walls(row, col, value)
        self._remove_neighbor_walls(row, col, ~value)

    def add_walls(self, row: int, col: int, walls: Walls) -> None:
        """Add walls to the maze."""
        self._cells[self._index(row, col)] |= walls.value
        self._add_neighbor_walls(row, col, walls)

    def _add_neighbor_walls(self, row: int, col: int, walls: Walls) -> None:
        """Add walls to the neighboring cells in the maze."""
        for added in walls:
            match added:
                case Walls.NORTH if row > 0:
                    self._cells[self._index(row - 1, col)] |= Walls.SOUTH.value
                case Walls.EAST if col < self.width - 1:
                    self._cells[self._index(row, col + 1)] |= Walls.WEST.value
                case Walls.SOUTH if row < self.height - 1:
                    self._cells[self._index(row + 1, col)] |= Walls.NORTH.value
                case Walls.WEST if col > 0:
                    self._cells[self._index(row, col - 1)] |= Walls.EAST.value

    def remove_walls(self, row: int, col: int, walls: Walls) -> None:
        """Remove walls from the maze."""
        self._cells[self._index(row, col)] &= (~walls).value
        self._remove_neighbor_walls(row, col, walls)

    def _remove_neighbor_walls(self, row: int, col: int, walls: Walls) -> None:
        """Remove walls from the neighboring cells in the maze."""
        for removed in walls:
            match removed:
                case Walls.NORTH:
                    if row == 0:
                        raise ValueError("cannot remove the NORTH wall from the top row")
                    self._cells[self._index(row - 1, col)] &= (~Walls.SOUTH).value
                case Walls.EAST:
                    if col == self.width - 1:
                        raise ValueError("cannot remove the EAST wall from the rightmost column")
                    self._cells[self._index(row, col + 1)] &= (~Walls.WEST).value
                case Walls.SOUTH:
                    if row == self.height - 1:
                        raise ValueError("cannot remove the SOUTH wall from the bottom row")
                    self._cells[self._index(row + 1, col)] &= (~Walls.NORTH).value
                case Walls.WEST:
                    if col == 0:
                        raise ValueError("cannot remove the WEST wall from the leftmost column")
                    self._cells[self._index(row, col - 1)] &= (~Walls.EAST).value

    def __iter__(self) -> Iterator[tuple[int, int, Walls]]:
        """Iterate over the cells in the maze, with indexes.

        Returns:
            Iterator: An iterator over the cells and their indexes.

        Yields:
            (int, int, Walls): A (row, col, walls) tuple.
        """
        return ((*divmod(idx, self.width), Walls(cell)) for idx, cell in enumerate(self._cells))

    # A maze is not a container: ``Walls.NORTH in maze`` should not work.
    __contains__ = None

    def render_screen(
            self,
            charset: Charset = ascii_charset,
            cell_width: int = 3,
            cell_height: int = 1,
            force_corners: bool = True,
    ) -> np.ndarray[tuple[int, int], np.dtypes.StrDType]:
        """Render the maze as text (in a numpy 2D array)"""
        screen = np.array([[' ' for _ in range(self.width * (cell_width + 1) + 1)] for _ in range(self.height * (cell_height + 1) + 1)])
        bottom_row = self.height * (cell_height + 1)
        rightmost_col = self.width * (cell_width + 1)
        for row in range(self.height):
            top_row = row * (cell_height + 1)
            for col in range(self.width):
                left_col = col * (cell_width + 1)
                cell = self[row, col]
                corner = LineDirection.EMPTY
                if row > 0 and (self[row - 1, col] & Walls.WEST or force_corners):
                    corner |= LineDirection.UP
                if col > 0 and (self[row, col - 1] & Walls.NORTH or force_corners):
                    corner |= LineDirection.LEFT
                if cell & Walls.NORTH or force_corners:
                    corner |= LineDirection.RIGHT
                if cell & Walls.WEST or force_corners:
                    corner |= LineDirection.DOWN
                # corner = (
                #     (LineDirection.UP if row > 0 and self[row - 1, col] & Walls.WEST else LineDirection.EMPTY) |
                #     (LineDirection.LEFT if col > 0 and self[row, col - 1] & Walls.NORTH else LineDirection.EMPTY) |
                #     (LineDirection.RIGHT if cell & Walls.NORTH else LineDirection.EMPTY) |
                #     (LineDirection.DOWN if cell & Walls.WEST else LineDirection.EMPTY)
                # )
                screen[top_row, left_col] = charset(corner)
                screen[
                    top_row, left_col + 1:left_col + 1 + cell_width
                ] = charset(LineDirection.HORIZONTAL if cell & Walls.NORTH else LineDirection.EMPTY)
                screen[
                    top_row + 1:top_row + 1 + cell_height, left_col
                ] = charset(LineDirection.VERTICAL if cell & Walls.WEST else LineDirection.EMPTY)
                if row == self.height - 1:
                    corner = LineDirection.RIGHT
                    if col > 0:
                        corner |= LineDirection.LEFT
                    if cell & Walls.WEST or force_corners:
                        corner |= LineDirection.UP
                    screen[bottom_row, left_col] = charset(corner)
                    screen[bottom_row, left_col + 1:left_col + 1 + cell_width] = charset(LineDirection.HORIZONTAL)

            corner = LineDirection.DOWN
            if row > 0:
                corner |= LineDirection.UP
            if cell & Walls.NORTH or force_corners:
                corner |= LineDirection.LEFT
            screen[top_row, rightmost_col] = charset(corner)
            screen[top_row + 1:top_row + 1 + cell_height, rightmost_col] = charset(LineDirection.VERTICAL)
            if row == self.height - 1:
                screen[bottom_row, rightmost_col] = charset(LineDirection.UP | LineDirection.LEFT)
        return screen

    def render(self, charset: Charset = ascii_charset, cell_width: int = 3, cell_height: int = 1, force_corners: bool = True) -> str:
        """Render the maze as text"""
        return '\n'.join(''.join(row) for row in self.render_screen(charset, cell_width, cell_height, force_corners)) + '\n'

    def _validate(self) -> None:
        """Raises an exception if the maze is not enclosed by walls or if 2 adjacent cells disagree on their shared wall"""
        for idx, cell in enumerate(self._cells):
            walls = Walls(cell)
            row, col = divmod(idx, self.width)

            if Walls.NORTH not in walls:
                if row == 0:
                    raise ValueError(f"missing NORTH wall at ({row}, {col})")
                if Walls.SOUTH in Walls(self._cells[idx - self.width]):
                    raise ValueError(f"disagreeing cells ({row}, {col}) and ({row - 1}, {col})")

            if Walls.EAST not in walls:
                if col == self.width - 1:
                    raise ValueError(f"missing EAST wall at ({row}, {col})")
                if Walls.WEST in Walls(self._cells[idx + 1]):
                    raise ValueError(f"disagreeing cells ({row}, {col}) and ({row}, {col + 1})")

            if Walls.SOUTH not in walls:
                if row == self.height - 1:
                    raise ValueError(f"missing SOUTH wall at ({row}, {col})")
                # if Walls.NORTH in Walls(self._cells[idx + self.width]):
                #     raise ValueError(f"disagreeing cells ({row}, {col}) and ({row + 1}, {col})")

            if Walls.WEST not in walls:
                if col == 0:
                    raise ValueError(f"missing WEST wall at ({row}, {col})")
                # if Walls.EAST in Walls(self._cells[idx - 1]):
                #     raise ValueError(f"disagreeing cells ({row}, {col}) and ({row}, {col - 1})")


@dataclass(kw_only=True, slots=True)
class ExtraCellInfo(NumpyCheat):
    """Holds additional information per cell managed by the robot/simulation."""
    weight: float | None = None
    color: tuple[int, int, int] | str | None = None  # Maybe a better type here
    visited: int = 0

    def visit_cell(self) -> None:
        """Increase the visit counter for the cell."""
        self.visited += 1

    def reset_color_if(self, color: tuple[int, int, int] | str | None = None) -> None:
        """Reset the color field to ``None``.

        Args:
            color (tuple[int, int, int] | str | None, optional):
                If not ``None``, reset the color field only if it is equal to the provided color.
                Defaults to None.
        """
        if color is None or self.color == color:
            self.color = None


if TYPE_CHECKING:
    type Step = tuple[int, int]
    type Route = list[Step]
    type RouteLike = Iterable[Step] | None


class ExtendedMaze(Maze):
    """A class representing a maze with additional solving information."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._connectivity: UnionFind[tuple[int, int]] | None = None
        self._route: Route = []
        self._extra_info: npt.NDArray[ExtraCellInfo] = np.empty(self.size, dtype=ExtraCellInfo)
        self._curr_gen = 0
        self._prev_gen = self._curr_gen - 1
        self.reset_info()

    def reset_info(self) -> None:
        """Reset the extra info held in the maze."""
        for row in range(self.height):
            for col in range(self.width):
                self._extra_info[row, col] = ExtraCellInfo()

    @property
    def extra_info(self) -> npt.NDArray[ExtraCellInfo]:
        """Return an array-like for storing extra info about cells in the maze."""
        return self._extra_info

    @extra_info.deleter
    def extra_info(self) -> None:
        """Reset the extra info."""
        self.reset_info()

    def iter_info(self) -> Iterator[tuple[int, int, ExtraCellInfo]]:
        """Iterate over the info of the cells in the maze, with indexes.

        Returns:
            Iterator: An iterator over the cells' infos and their indexes.

        Yields:
            (int, int, ExtraCellInfo): A (row, col, info) tuple.
        """
        return ((row, col, self._extra_info[row, col]) for row in range(self.height) for col in range(self.width))

    def iter_all(self) -> Iterator[tuple[int, int, Walls, ExtraCellInfo]]:
        """Iterate over the cells in the maze, with indexes and info.

        Returns:
            Iterator: An iterator over the cells, their indexes and their info.

        Yields:
            (int, int, Walls, ExtraCellInfo): A (row, col, walls, info) tuple.
        """
        return ((row, col, Walls(cell), info) for cell, (row, col, info) in zip(self._cells, self.iter_info()))

    @property
    def route(self) -> Route:
        """The route for the maze, in (row, col) pairs."""
        return self._route

    @route.setter
    def route(self, value: RouteLike) -> None:
        """Set the route for the maze, in (row, col) pairs."""
        self._route = list(value or ())

    @route.deleter
    def route(self) -> None:
        """Reset the current route."""
        self._route = []

    def explored_cells_count(self) -> int:
        """Calculate the number of cells that were visited.

        Returns:
            int: The number of cells that were visited.
        """
        return sum(info.visited > 0 for _, _, info in self.iter_info())

    def explored_cells_percentage(self) -> float:
        """Calculate the percentage of cells that were visited.

        Returns:
            float: The percentage of cells that were visited.
        """
        return self.explored_cells_count() / self.cell_size

    @property
    def connectivity(self) -> UnionFind[tuple[int, int]]:
        """Calculate connected groups in the maze.

        Returns:
            UnionFind[tuple[int, int]]: All connected groups of cells.
        """
        if self._connectivity is None:
            self._connectivity = UnionFind()
            for row, col, walls in self:
                self._connectivity.find(cell := (row, col))
                for missing in ~walls:
                    match missing:
                        case Walls.NORTH:
                            self._connectivity.union(cell, (row - 1, col))
                        case Walls.EAST:
                            self._connectivity.union(cell, (row, col + 1))
                        case Walls.SOUTH:
                            self._connectivity.union(cell, (row + 1, col))
                        case Walls.WEST:
                            self._connectivity.union(cell, (row, col - 1))
        return self._connectivity

    @connectivity.deleter
    def connectivity(self) -> None:
        """Clear connectivity caching."""
        self._connectivity = None

    def __setitem__(self, idx: tuple[int, int], value: Walls) -> None:
        old_val = self[idx]
        try:
            return super().__setitem__(idx, value)
        finally:
            if old_val != self[idx]:
                self.mark_changed()

    def add_walls(self, row: int, col: int, walls: Walls) -> None:
        old_val = self[row, col]
        try:
            return super().add_walls(row, col, walls)
        finally:
            if old_val != self[row, col]:
                self.mark_changed()

    def remove_walls(self, row: int, col: int, walls: Walls) -> None:
        old_val = self[row, col]
        try:
            return super().remove_walls(row, col, walls)
        finally:
            if old_val != self[row, col]:
                self.mark_changed()

    def mark_changed(self) -> None:
        """Mark that the maze has changed, regardless of wall changes."""
        del self.connectivity
        self._curr_gen = self._prev_gen + 1

    def changed(self) -> bool:
        """Check whether the maze has changed (added/removed walls) since the last call.

        Returns:
            bool: True if there was a change since the last call, otherwise False.
        """
        prev, self._prev_gen = self._prev_gen, self._curr_gen
        return prev != self._curr_gen

    __ROBOT_MARKERS: dict[Direction, str] = {
        Direction.NORTH: '^',
        Direction.EAST: '>',
        Direction.SOUTH: 'v',
        Direction.WEST: '<',
    }

    def render_extra(
            self,
            *,
            charset: Charset = ascii_charset,
            pos: tuple[int, int, Direction] | None = None,
            goals: Set[tuple[int, int]] = frozenset(),
            weights: bool = True,
    ) -> str:
        """Render the maze with extra information"""
        cell_width = cell_height = 1
        if weights:
            cell_width = max(len(str(info.weight)) if info.weight else 1 for _, _, info in self.iter_info())
            cell_height = 2
        screen = self.render_screen(
            cell_width=2 + cell_width,
            cell_height=cell_height,
            charset=charset,
        )

        def screen_pos(row: int, col: int, /) -> tuple[int, int]:
            return (row + 1) * (cell_height + 1) - 1, col * (cell_width + 3) + 1 + ((cell_width + 1) // 2)

        if pos:
            screen[screen_pos(*pos[:2])] = self.__ROBOT_MARKERS.get(pos[2], 'S')
        for goal in goals - {(pos or ())[:2]}:
            screen[screen_pos(*goal)] = '@'

        def write_weight(screen_row: int, screen_left_col: int, weight: float | None, /) -> None:
            weight_str = f"{'' if weight is None else weight:>{cell_width}}"
            for i in range(cell_width):
                screen[screen_row, screen_left_col + i] = weight_str[i]

        if weights:
            for cell_row, cell_col, info in self.iter_info():
                write_weight(cell_row * (cell_height + 1) + 1, cell_col * (cell_width + 3) + 2, info.weight)
        return '\n'.join(''.join(row) for row in screen)
