"""Maze representation and utils

This module contains a class for representing mazes and tools for loading and
saving mazes to files.
"""

from __future__ import annotations

import math
import os
import re

from enum import Flag, IntEnum
from functools import reduce
from operator import or_
from typing import cast, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from os import PathLike
    from typing import BinaryIO, Callable, Iterator, Literal, Self, TextIO, TypeAlias

MazeSize: TypeAlias = "tuple[int, int]"


class Direction(IntEnum):
    """Bit masks for the directions."""
    NORTH = 0x1
    EAST = 0x2
    SOUTH = 0x4
    WEST = 0x8
    NORTH_EAST = NORTH | EAST
    NORTH_WEST = NORTH | WEST
    SOUTH_EAST = SOUTH | EAST
    SOUTH_WEST = SOUTH | WEST

    def turn_left(self) -> Direction:  # pylint: disable=too-many-return-statements
        """Return the direction that is the result of turning left (90 degrees counter-clockwise)."""
        match self:
            case Direction.NORTH: return Direction.WEST
            case Direction.EAST: return Direction.NORTH
            case Direction.SOUTH: return Direction.EAST
            case Direction.WEST: return Direction.SOUTH
            case Direction.NORTH_EAST: return Direction.NORTH_WEST
            case Direction.NORTH_WEST: return Direction.SOUTH_WEST
            case Direction.SOUTH_EAST: return Direction.NORTH_EAST
            case Direction.SOUTH_WEST: return Direction.SOUTH_EAST

    def turn_right(self) -> Direction:  # pylint: disable=too-many-return-statements
        """Return the direction that is the result of turning right (90 degrees clockwise)."""
        match self:
            case Direction.NORTH: return Direction.EAST
            case Direction.EAST: return Direction.SOUTH
            case Direction.SOUTH: return Direction.WEST
            case Direction.WEST: return Direction.NORTH
            case Direction.NORTH_EAST: return Direction.SOUTH_EAST
            case Direction.NORTH_WEST: return Direction.NORTH_EAST
            case Direction.SOUTH_EAST: return Direction.SOUTH_WEST
            case Direction.SOUTH_WEST: return Direction.NORTH_WEST

    def turn_back(self) -> Direction:  # pylint: disable=too-many-return-statements
        """Return the direction that is the result of turning back (180 degrees)."""
        match self:
            case Direction.NORTH: return Direction.SOUTH
            case Direction.EAST: return Direction.WEST
            case Direction.SOUTH: return Direction.NORTH
            case Direction.WEST: return Direction.EAST
            case Direction.NORTH_EAST: return Direction.SOUTH_WEST
            case Direction.NORTH_WEST: return Direction.SOUTH_EAST
            case Direction.SOUTH_EAST: return Direction.NORTH_WEST
            case Direction.SOUTH_WEST: return Direction.NORTH_EAST

    def half_turn_left(self) -> Direction:  # pylint: disable=too-many-return-statements
        """Return the direction that is the result of turning left (90 degrees counter-clockwise)."""
        match self:
            case Direction.NORTH: return Direction.NORTH_WEST
            case Direction.EAST: return Direction.NORTH_EAST
            case Direction.SOUTH: return Direction.SOUTH_EAST
            case Direction.WEST: return Direction.SOUTH_WEST
            case Direction.NORTH_EAST: return Direction.NORTH
            case Direction.NORTH_WEST: return Direction.WEST
            case Direction.SOUTH_EAST: return Direction.EAST
            case Direction.SOUTH_WEST: return Direction.SOUTH

    def half_turn_right(self) -> Direction:  # pylint: disable=too-many-return-statements
        """Return the direction that is the result of turning right (90 degrees clockwise)."""
        match self:
            case Direction.NORTH: return Direction.NORTH_EAST
            case Direction.EAST: return Direction.SOUTH_EAST
            case Direction.SOUTH: return Direction.SOUTH_WEST
            case Direction.WEST: return Direction.NORTH_WEST
            case Direction.NORTH_EAST: return Direction.EAST
            case Direction.NORTH_WEST: return Direction.NORTH
            case Direction.SOUTH_EAST: return Direction.SOUTH
            case Direction.SOUTH_WEST: return Direction.WEST

    def __or__(self, other: Self | int) -> Self:
        return type(self)(super().__or__(other))

    def __and__(self, other: Self | int) -> Self:
        return type(self)(super().__and__(other))

    def __xor__(self, other: Self | int) -> Self:
        return type(self)(super().__xor__(other))

    def __str__(self):
        return self.name


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
    def from_maz_file(cls, maz: PathLike | BinaryIO, size: MazeSize | None = None) -> Self:
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
        if isinstance(maz, (os.PathLike, str, bytes)):
            maz = open(maz, "rb")

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
    def from_num_file(cls, num_file: PathLike | TextIO, size: MazeSize | None = None) -> Self:
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

        if isinstance(num_file, (os.PathLike, str, bytes)):
            num_file = open(num_file, "rt", encoding="ASCII")

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
    def from_maze_file(cls, maze_file: PathLike | TextIO, cell_height: int = 1, cell_width: int = 3, empty: str = ' ') -> Self:
        """
        Load a maze from a file or file-like object.

        Format:
            A text drawing of the maze.
            TODO: add format explanation and links
        """
        if isinstance(maze_file, (os.PathLike, str, bytes)):
            maze_file = open(maze_file, "rt", encoding="ASCII")

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
    def from_csv_file(cls, csv_file: PathLike | TextIO) -> Self:
        """
        Load a maze from a csv file or file-like object.

        Format:
            A csv where each row has the cells for the corresponding row
            TODO: add format explanation and links
        """
        if isinstance(csv_file, (os.PathLike, str, bytes)):
            csv_file = open(csv_file, "rt", encoding="ASCII")

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
    def from_file(cls, maze_file: PathLike, fmt: Literal['maz', 'num', 'csv', 'maze', None] = None) -> Self:
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
        cell = self._index(row, col)
        if cell >= self.cell_size:
            return default
        return Walls(self._cells[cell])

    def __getitem__(self, idx: tuple[int, int]) -> Walls:
        if not isinstance(idx, tuple) or len(idx) != 2 or not all(isinstance(i, int) for i in idx):
            raise TypeError(f"expected 2 ints: (row, col), got {idx!r}")
        row, col = idx
        # TODO: make this prettier
        if row >= self.height:
            raise IndexError("TODO: pretty string")
        if col >= self.width:
            raise IndexError("TODO: pretty string")
        val = self.get(row, col, _Missing())
        assert not isinstance(val, _Missing)
        return val

    def __setitem__(self, idx: tuple[int, int], value: Walls):
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

    def add_walls(self, row: int, col: int, walls: Walls):
        """Add walls to the maze."""
        self._cells[self._index(row, col)] |= walls.value
        self._add_neighbor_walls(row, col, walls)

    def _add_neighbor_walls(self, row: int, col: int, walls: Walls):
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

    def remove_walls(self, row: int, col: int, walls: Walls):
        """Remove walls from the maze."""
        self._cells[self._index(row, col)] &= (~walls).value
        self._remove_neighbor_walls(row, col, walls)

    def _remove_neighbor_walls(self, row: int, col: int, walls: Walls):
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

    def render_lines(
            self,
            charset: Charset = ascii_charset,
            cell_width: int = 3,
            cell_height: int = 1,
            force_corners: bool = True,
    ) -> list[str]:
        """Render the maze as text (in separate lines)"""
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
        return [''.join(row) for row in screen]

    def render(self, charset: Charset = ascii_charset, cell_width: int = 3, cell_height: int = 1, force_corners: bool = True) -> str:
        """Render the maze as text"""
        return '\n'.join(self.render_lines(charset, cell_width, cell_height, force_corners)) + '\n'

    def _validate(self):
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
