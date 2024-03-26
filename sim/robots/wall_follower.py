"""wall follower robot

Uses the "classic" follow the right/left wall to leave a maze.
This robot will be unable to find certain goals (floaters).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from .utils import Action, direction_to_wall, turns_for_rel_direction
from . import utils  # for ENABLE_VICTORY_DANCE
from ..directions import RelativeDirection

if TYPE_CHECKING:
    from typing import Literal
    from .utils import Algorithm, Robot
    from ..maze import Maze


def _wall_follower_robot(
        maze: Maze,
        goals: set[tuple[int, int]],
        *,
        follow: Literal[RelativeDirection.LEFT, RelativeDirection.RIGHT],
) -> Robot:
    """A robot that follows the wall.

    Returns:
        Robot: The robot's brain.
    """
    destination = set(goals)

    match follow:
        case RelativeDirection.LEFT | RelativeDirection.RIGHT: pass
        case RelativeDirection(): raise ValueError(f"invalid follow direction: {follow}")
        case _: raise TypeError(f"invalid follow type: {type(follow)}")
    next_direction = follow.invert()

    pos_row, pos_col, facing = yield Action.READY

    while (pos_row, pos_col) not in destination:
        walls = maze[pos_row, pos_col]
        if direction_to_wall(turn := facing.turn(follow)) not in walls:
            rel = follow
        elif direction_to_wall(turn := turn.turn(next_direction)) not in walls:
            assert turn == facing, "turned back but didn't return"
            rel = RelativeDirection.FRONT
        elif direction_to_wall(turn := turn.turn(next_direction)) not in walls:
            rel = next_direction
        elif direction_to_wall(turn := turn.turn(next_direction)) not in walls:
            rel = RelativeDirection.BACK
        else:
            # we're in a box...
            return

        for turn_action in turns_for_rel_direction(rel, follow):
            r, c, facing = yield turn_action
            assert (r, c) == (pos_row, pos_col), "moved while turning"
            assert maze[r, c] == walls, "walls changed while turning"
        assert facing == turn, "turning failed"

        pos_row, pos_col, facing = yield Action.FORWARD

    # Victory spin
    while utils.ENABLE_VICTORY_DANCE:
        yield Action.from_rel_direction(follow)


def wall_follower_robot(follow: Literal[RelativeDirection.LEFT, RelativeDirection.RIGHT]) -> Algorithm:
    """A robot that follows the wall.

    Returns:
        Robot: The robot's brain.
    """
    def _inner(maze: Maze, goals: set[tuple[int, int]]) -> Robot:
        return _wall_follower_robot(maze, goals, follow=follow)

    return _inner
