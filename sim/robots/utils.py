"""generic robot utilities

Utility functions for all robots.
"""

from __future__ import annotations

import os
import random

from enum import auto, Enum
from typing import NamedTuple, overload, TYPE_CHECKING

from ..maze import Direction, ExtendedMaze, Maze, RelativeDirection, Walls, PRIMARY_DIRECTIONS

if TYPE_CHECKING:
    from collections.abc import Iterable, Generator, Set
    from typing import Callable, Literal

ENABLE_VICTORY_DANCE = os.environ.get('MICROMOUSE_VICTORY_DANCE', 'n') == 'y'

# TODO: replace prints with logging


class Action(Enum):
    """TODO: docs"""
    READY = auto()
    RESET = auto()
    FORWARD = auto()
    BACKWARDS = auto()
    TURN_LEFT = auto()
    TURN_RIGHT = auto()

    @overload
    @classmethod
    def from_rel_direction(cls, direction: Literal[RelativeDirection.FRONT]) -> Literal[Action.FORWARD]: ...
    @overload
    @classmethod
    def from_rel_direction(cls, direction: Literal[RelativeDirection.BACK]) -> Literal[Action.BACKWARDS]: ...
    @overload
    @classmethod
    def from_rel_direction(cls, direction: Literal[RelativeDirection.LEFT]) -> Literal[Action.TURN_LEFT]: ...
    @overload
    @classmethod
    def from_rel_direction(cls, direction: Literal[RelativeDirection.RIGHT]) -> Literal[Action.TURN_RIGHT]: ...

    @classmethod
    def from_rel_direction(cls, direction: RelativeDirection) -> Action:
        """Create an action from a relative direction."""
        match direction:
            case RelativeDirection.FRONT: return cls.FORWARD
            case RelativeDirection.BACK: return cls.BACKWARDS
            case RelativeDirection.LEFT: return cls.TURN_LEFT
            case RelativeDirection.RIGHT: return cls.TURN_RIGHT


def turns_for_rel_direction(
    direction: RelativeDirection,
    follow: Literal[RelativeDirection.LEFT, RelativeDirection.RIGHT] = RelativeDirection.LEFT,
) -> list[Literal[Action.TURN_LEFT, Action.TURN_RIGHT]]:
    """Create the turns needed to face in a relative direction."""
    match direction:
        case RelativeDirection.FRONT: return []
        case RelativeDirection.BACK: return [Action.from_rel_direction(follow.invert())] * 2
        case RelativeDirection.LEFT: return [Action.TURN_LEFT]
        case RelativeDirection.RIGHT: return [Action.TURN_RIGHT]


def needed_turns_for_rel_direction(
    direction: RelativeDirection,
    turn_action: Literal[Action.TURN_LEFT, Action.TURN_RIGHT],
) -> int:
    """Get the number of turns needed to face in a relative direction using only one turn type."""
    match direction:
        case RelativeDirection.FRONT: return 0
        case RelativeDirection.BACK: return 2
        case RelativeDirection.LEFT: return 3 if turn_action is Action.TURN_RIGHT else 1
        case RelativeDirection.RIGHT: return 3 if turn_action is Action.TURN_LEFT else 1


class RobotState(NamedTuple):
    """TODO: docs

    Represents the current robot's state
    """
    row: int
    col: int
    facing: Direction


if TYPE_CHECKING:
    type Robot = Generator[Action, RobotState, None]
    type Algorithm = Callable[[ExtendedMaze, set[tuple[int, int]]], Robot]


def _wall_to_direction(wall: Walls) -> Direction:
    match wall:
        case Walls.NORTH | Walls.EAST | Walls.SOUTH | Walls.WEST:
            return Direction(wall.value)
        case _:
            raise ValueError(f"can only convert a single wall (not {wall!r})")


def walls_to_directions(walls: Walls) -> list[Direction]:
    """TODO: docs (converts a cell's wall spec into a sorted list of available directions)"""
    return sorted(_wall_to_direction(missing_wall) for missing_wall in ~walls)


def direction_to_wall(direction: Direction) -> Walls:
    """TODO: docs (converts a direction spec into a wall)"""
    match direction:
        case Direction.NORTH: return Walls.NORTH
        case Direction.EAST: return Walls.EAST
        case Direction.SOUTH: return Walls.SOUTH
        case Direction.WEST: return Walls.WEST
        case _: raise ValueError(f"can only convert the primary directions (not {direction})")


def _adjacent_cells_impl(maze: Maze, cells: Iterable[tuple[int, int]]) -> Iterable[tuple[int, int]]:
    for row, col in cells:
        walls = maze[row, col]
        if Walls.NORTH not in walls:
            yield (row - 1, col)
        if Walls.EAST not in walls:
            yield (row, col + 1)
        if Walls.SOUTH not in walls:
            yield (row + 1, col)
        if Walls.WEST not in walls:
            yield (row, col - 1)


def adjacent_cells(maze: Maze, cells: Iterable[tuple[int, int]], without: set[tuple[int, int]] | None = None) -> set[tuple[int, int]]:
    """
    Returns a cell wil all cells that are adjacent (without diagonals) to a
    cell in ``cells`` and don't have a wall separating them from their relevant
    cell in ``cells``, excluding the ``without`` cells.

    Args:
        maze (Maze): The maze.
        cells (Iterable[tuple[int, int]]): The cells to find adjacent cells of.
        without (set[tuple[int, int]] | None, optional): Cells to exclude from the result. Defaults to None.

    Returns:
        set[tuple[int, int]]: A cell with all relevant adjacent cells.
    """
    return set(_adjacent_cells_impl(maze, cells)) - (without or set())


def direction_to_cell(cell: tuple[int, int], direction: Direction) -> tuple[int, int]:
    """
    Returns the indexes of the cell to the ``direction`` of the given cell.

    Args:
        cell (tuple[int, int]): The current cell.
        direction (Direction): The direction to move at. Must be a primary direction.

    Raises:
        ValueError: ``direction`` is a secondary direction.

    Returns:
        tuple[int, int]: The required cell, may be out of bounds of the maze.
    """
    row, col = cell
    match direction:
        case Direction.NORTH:
            return (row - 1, col)
        case Direction.EAST:
            return (row, col + 1)
        case Direction.SOUTH:
            return (row + 1, col)
        case Direction.WEST:
            return (row, col - 1)
    raise ValueError(f"unsupported direction {direction}")


def cell_to_direction(  # pylint: disable=too-many-return-statements
        src_cell: tuple[int, int],
        dst_cell: tuple[int, int],
        diagonals: bool = False,
) -> Direction:
    """
    Returns the direction to move at to get from ``src_cell`` to the adjacent ``dst_cell``.

    Args:
        src_cell (tuple[int, int]): The source cell.
        dst_cell (tuple[int, int]): The destination cell.
        diagonals (bool, optional): Whether diagonals are allowed. Defaults to False.

    Raises:
        ValueError: If ``dst_cell`` is not adjacent to ``src_cell``.

    Returns:
        Direction: The direction that will return ``dst_cell`` from a call to direction_to_cell.

    >>> cell = (4, 5)
    >>> direction = cell_to_direction(cell, (4, 6))
    >>> direction_to_cell(cell, direction)
    (4, 6)
    """
    src_row, src_col = src_cell
    dst_row, dst_col = dst_cell
    match (dst_row - src_row, dst_col - src_col):
        case (-1, 0):
            return Direction.NORTH
        case (0, 1):
            return Direction.EAST
        case (1, 0):
            return Direction.SOUTH
        case (0, -1):
            return Direction.WEST
        case (-1, 1) if diagonals:
            return Direction.NORTH_EAST
        case (1, 1) if diagonals:
            return Direction.SOUTH_EAST
        case (1, -1) if diagonals:
            return Direction.SOUTH_WEST
        case (-1, -1) if diagonals:
            return Direction.NORTH_WEST
    raise ValueError(f"{dst_cell} is not adjacent to {src_cell} ({diagonals=})")


def abs_turn_to_actions(before: Direction, after: Direction, allow_reverse: bool = True) -> list[Action]:
    """Actions required to get from a cell, facing ``before`` to the cell at the ``after`` direction."""
    if before == after:
        return [Action.FORWARD]
    if before.turn_back() == after:
        return [Action.BACKWARDS] if allow_reverse else [Action.TURN_LEFT, Action.TURN_LEFT, Action.FORWARD]
    if before.turn_left() == after:
        return [Action.TURN_LEFT, Action.FORWARD]
    if before.turn_right() == after:
        return [Action.TURN_RIGHT, Action.FORWARD]
    raise ValueError(f"Turn not supported: {before} -> {after}")


def abs_turn_to_rel(before: Direction, after: Direction) -> RelativeDirection:
    """Relative direction to get from a facing ``before`` to facing ``after`` direction."""
    if before == after:
        return RelativeDirection.FRONT
    if before.turn_back() == after:
        return RelativeDirection.BACK
    if before.turn_left() == after:
        return RelativeDirection.LEFT
    if before.turn_right() == after:
        return RelativeDirection.RIGHT
    raise ValueError(f"Turn not supported: {before} -> {after}")


type Vertex = tuple[int, int, Direction]


def _callable_weights(
        weights: dict[RelativeDirection, float] | Callable[[RelativeDirection], float],
) -> Callable[[RelativeDirection], float]:
    if isinstance(weights, dict):
        if len(weights) != len(RelativeDirection):
            raise ValueError(f"missing weights: {', '.join(map(str, set(RelativeDirection) - set(weights)))}")

        return weights.__getitem__
    return weights


def build_weighted_graph(
        maze: ExtendedMaze,
        weights: dict[RelativeDirection, float] | Callable[[RelativeDirection], float],
        *,
        order: int = 1,
        diagonals: bool = False,
        without: Set[tuple[int, int]] = frozenset(),
) -> dict[Vertex, dict[Vertex, float]]:
    """Build a graph from"""
    if order <= 0:
        raise ValueError(f"order must be positive: {order}")
    assert order == 1, "Higher orders not yet supported"
    assert not diagonals, "Diagonals not yet supported"

    weights = _callable_weights(weights)
    graph = {}

    for row, col, walls in maze:
        if (row, col) in without:
            continue

        # for dist in range(1, order+1):
        for direction in walls_to_directions(walls):
            dest = direction_to_cell((row, col), direction)
            if dest in without:
                continue
            graph.setdefault((row, col, direction), {})[*dest, direction.turn_back()] = 0

        for d1 in PRIMARY_DIRECTIONS:
            v1 = row, col, d1
            if v1 not in graph:
                continue
            for d2 in PRIMARY_DIRECTIONS:
                if d2 == d1:
                    continue
                v2 = row, col, d2
                if v2 not in graph:
                    continue
                # when arriving from ``d1`` wall, we are facing ``d1.turn_back()``
                graph[v1][v2] = weights(abs_turn_to_rel(d1.turn_back(), d2))

    return graph


def identity[T](obj: T) -> T:
    """Return ``obj``.

    Args:
        obj (T): Any python object.

    Returns:
        T: ``obj``.

    >>> x = [1, 2, 3]
    >>> identity(x) is x
    True

    >>> x = object()
    >>> identity(x) is x
    True
    """
    return obj


def shuffled[T](lst: list[T]) -> list[T]:
    """Shuffle the list (in-place) and return it.

    Args:
        lst (list[T]): The list to shuffle.

    Returns:
        list[T]: ``lst`` after shuffling.

    >>> lst = [1, 2, 3]
    >>> shuffled(lst) is lst
    True
    """
    random.shuffle(lst)
    return lst
