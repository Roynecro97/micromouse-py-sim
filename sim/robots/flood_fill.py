"""flood fill robot

Uses flood-fill to get to the goal and then back to the start,
then uses an optimal route.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .utils import Action, adjacent_cells, shuffled, walls_to_directions
from ..maze import Direction

if TYPE_CHECKING:
    from ..maze import ExtendedMaze, ExtraCellInfo
    from .utils import Robot


def _simple_flood_fill(maze: ExtendedMaze, goals: set[tuple[int, int]]) -> Robot:
    """A robot that solves the maze using a simple implementation of the flood-fill algorithm.

    Returns:
        Robot: The robot's brain.
    """
    def _calc_flood_fill():
        seen = set()
        current = goals
        marker = 0
        while True:
            for cell in current:
                info: ExtraCellInfo = maze.extra_info[cell]
                info.weight = marker
            marker += 1
            seen.update(current)
            current = adjacent_cells(maze, current, seen)
            if len(seen) >= maze.cell_size:
                break
        assert len(seen) == maze.cell_size, "new cells created"

    def _visual_flood():  # TODO: delete once we have stat representation on our GUI
        import numpy as np  # pylint: disable=import-outside-toplevel

        def _weight(info: ExtraCellInfo) -> float:
            return info.weight if info.weight is not None else -1

        return np.frompyfunc(_weight, nin=1, nout=1)(maze.extra_info)

    def _direction_to_cell(direction: Direction) -> tuple[int, int]:
        match direction:
            case Direction.NORTH: return (pos_row - 1, pos_col)
            case Direction.EAST: return (pos_row, pos_col + 1)
            case Direction.SOUTH: return (pos_row + 1, pos_col)
            case Direction.WEST: return (pos_row, pos_col - 1)
        raise ValueError(f"unsupported direction {direction}")

    def _priority(direction: Direction) -> tuple[float, int]:
        cell = maze.extra_info[_direction_to_cell(direction)]
        return cell.weight, cell.visited

    _calc_flood_fill()  # for simulation rendering before movement starts - completely redundant

    pos_row, pos_col, facing = yield Action.READY
    maze.route.append((pos_row, pos_col))

    while (pos_row, pos_col) not in goals:
        maze.extra_info[pos_row, pos_col].visit_cell()
        _calc_flood_fill()
        print(f"floodmouse: flooded ->\n{_visual_flood()}")
        walls = maze[pos_row, pos_col]
        print(f"floodmouse: at {(pos_row, pos_col)} facing {facing} with {walls}")
        new_direction = min(
            shuffled(walls_to_directions(walls)),  # regular / reversed / random
            key=_priority,
        )
        print(f"floodmouse: chose to flood {new_direction}")
        if new_direction == facing:
            print("floodmouse: will move forward")
            action = Action.FORWARD
        elif new_direction == facing.turn_back():
            print("floodmouse: will move in reverse")
            action = Action.BACKWARDS
        else:
            if new_direction == facing.turn_left():
                print("floodmouse: turning left")
                turn_action = Action.TURN_LEFT
            elif new_direction == facing.turn_right():
                print("floodmouse: turning right")
                turn_action = Action.TURN_RIGHT
            else:
                raise AssertionError(f"invalid turn from {facing} to {new_direction}")
            r, c, facing = yield turn_action
            assert (r, c) == (pos_row, pos_col), "moved while turning"
            assert maze[r, c] == walls, "walls changed while turning"
            assert facing == new_direction, "turning failed"
            print(f"floodmouse: now facing {facing}, will move forward")
            action = Action.FORWARD
        pos_row, pos_col, facing = yield action
        maze.route.append((pos_row, pos_col))


def simple_flood_fill(maze: ExtendedMaze, goals: set[tuple[int, int]]) -> Robot:
    """A robot that solves the maze using a simple implementation of the flood-fill algorithm.

    Returns:
        Robot: The robot's brain.
    """
    # Initial flood-fill to the goal
    yield from _simple_flood_fill(maze, goals)
    # Mark the starting point as the new goal and flood-fill to get there
    starting_pos = maze.route[0]
    maze.extra_info[starting_pos].color = "green"
    for goal in goals:
        maze.extra_info[goal].color = "blue"
    yield from _simple_flood_fill(maze, {starting_pos})
    # Reset colors, route and orientation (but keep walls, we should be at the starting position)
    maze.reset_info()
    del maze.route
    yield Action.RESET
    # Do the actual fast route
    yield from _simple_flood_fill(maze, goals)
