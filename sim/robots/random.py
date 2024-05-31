"""random robot

Moves to a random cell at every step.
"""

from __future__ import annotations

import random

from typing import TYPE_CHECKING
from . import utils  # for ENABLE_VICTORY_DANCE
from .utils import Action, walls_to_directions

if TYPE_CHECKING:
    from collections.abc import Set

    from .utils import Robot
    from ..maze import Maze


def random_robot(maze: Maze, goals: Set[tuple[int, int]]) -> Robot:
    """A robot with random movements.

    Returns:
        Robot: The robot's brain.
    """
    print(f"randomouse: starting in a {maze.height}x{maze.width} maze")
    print(f"randomouse: aiming for {goals}")
    pos_row, pos_col, heading = yield Action.READY
    print(f"randomouse: starting at {(pos_row, pos_col)} facing {heading}")

    while (pos_row, pos_col) not in goals:
        walls = maze[pos_row, pos_col]
        print(f"randomouse: at {(pos_row, pos_col)} facing {heading} with {walls}")
        new_direction = random.choice(walls_to_directions(walls))
        print(f"randomouse: chose to migrate {new_direction}")
        if new_direction == heading:
            print("randomouse: will move forward")
            action = Action.FORWARD
        elif new_direction == heading.turn_back():
            print("randomouse: will move in reverse")
            action = Action.BACKWARDS
        else:
            if new_direction == heading.turn_left():
                print("randomouse: turning left")
                turn_action = Action.TURN_LEFT
            elif new_direction == heading.turn_right():
                print("randomouse: turning right")
                turn_action = Action.TURN_RIGHT
            else:
                raise AssertionError(f"invalid turn from {heading} to {new_direction}")
            r, c, heading = yield turn_action
            assert (r, c) == (pos_row, pos_col), "randomouse: moved while turning"
            assert maze[r, c] == walls, "randomouse: walls changed while turning"
            assert heading == new_direction, "randomouse: turning failed"
            print(f"randomouse: now facing {heading}, will move forward")
            action = Action.FORWARD
        pos_row, pos_col, heading = yield action

    print("randomouse: victory")
    # Victory dance
    while utils.ENABLE_VICTORY_DANCE:
        yield random.choice((Action.TURN_LEFT, Action.TURN_RIGHT))
