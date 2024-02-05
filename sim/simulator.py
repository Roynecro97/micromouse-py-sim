"""Simulator
"""
from __future__ import annotations

import random

from enum import auto, Enum
from typing import NamedTuple, overload, TypeAlias, TYPE_CHECKING

from .maze import Direction, Maze, RelativeDirection, Walls

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Callable, Literal


class Action(Enum):
    """TODO: docs"""
    READY = auto()
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
    turn_action: Literal[Action.TURN_LEFT, Action.TURN_RIGHT, None] = None,
) -> list[Action]:
    """Create the turns needed to face in a relative direction."""
    match direction:
        case RelativeDirection.FRONT: return []
        case RelativeDirection.BACK: return [turn_action or Action.TURN_LEFT] * 2
        case RelativeDirection.LEFT if turn_action is Action.TURN_RIGHT: return [turn_action] * 3
        case RelativeDirection.LEFT: return [Action.TURN_LEFT]
        case RelativeDirection.RIGHT if turn_action is Action.TURN_LEFT: return [turn_action] * 3
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
    current_cell: Walls

# RobotState: TypeAlias = "tuple[int, int, Direction, Walls]"


Robot: TypeAlias = "Generator[Action, RobotState, None]"
Algorithm: TypeAlias = "Callable[[Maze, set[tuple[int, int]]], Robot]"


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
        case Direction.NORTH: return Walls.WEST
        case Direction.EAST: return Walls.NORTH
        case Direction.SOUTH: return Walls.EAST
        case Direction.WEST: return Walls.SOUTH
        case _: raise ValueError(f"can only convert the primary directions (not {direction})")


def random_robot(maze: Maze, goals: set[tuple[int, int]]) -> Robot:
    """A robot with random movements.

    Returns:
        Robot: The robot's brain.
    """
    destination = set(goals)
    pos_row, pos_col, facing, walls = yield Action.READY
    maze[pos_row, pos_col] = walls

    while (pos_row, pos_col) not in destination:
        new_direction = random.choice(walls_to_directions(walls))
        if new_direction == facing:
            action = Action.FORWARD
        elif new_direction == facing.turn_back():
            action = Action.BACKWARDS
        else:
            if new_direction == facing.turn_left():
                turn_action = Action.TURN_LEFT
            elif new_direction == facing.turn_right():
                turn_action = Action.TURN_RIGHT
            else:
                raise AssertionError(f"invalid turn from {facing} to {new_direction}")
            r, c, facing, w = yield turn_action
            assert (r, c) == (pos_row, pos_col), "moved while turning"
            assert w == walls, "walls changed while turning"
            assert facing == new_direction, "turning failed"
            action = Action.FORWARD
        pos_row, pos_col, facing, walls = yield action
        maze[pos_row, pos_col] = walls

    # # Victory dance
    # while True:
    #     yield random.choice((Action.TURN_LEFT, Action.TURN_RIGHT))


def _wall_follower_robot(maze: Maze, goals: set[tuple[int, int]], *
    , follow: Literal[RelativeDirection.LEFT, RelativeDirection.RIGHT]) -> Robot:
    """A robot that follows the wall.

    Returns:
        Robot: The robot's brain.
    """
    destination = set(goals)

    match follow:
        case RelativeDirection.LEFT: turn_action = Action.TURN_LEFT
        case RelativeDirection.RIGHT: turn_action = Action.TURN_RIGHT
        case RelativeDirection(): raise ValueError(f"invalid follow direction: {follow}")
        case _: raise TypeError(f"invalid follow type: {type(follow)}")
    next_direction = follow.invert()

    pos_row, pos_col, facing, walls = yield Action.READY
    maze[pos_row, pos_col] = walls

    while (pos_row, pos_col) not in destination:
        assert maze[pos_row, pos_col] is walls, "we're being lied to"
        if direction_to_wall(turn := facing.turn(follow)) not in walls:
            rel = follow
            # r, c, facing, w = yield turn_action
            # assert (r, c) == (pos_row, pos_col), "moved while turning"
            # assert w == walls, "walls changed while turning"
            # assert facing == turn, "turning failed"
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

        for _ in range(needed_turns_for_rel_direction(rel, turn_action)):
            r, c, facing, w = yield turn_action
            assert (r, c) == (pos_row, pos_col), "moved while turning"
            assert w == walls, "walls changed while turning"
        assert facing == turn, "turning failed"

        pos_row, pos_col, facing, walls = yield Action.FORWARD
        maze[pos_row, pos_col] = walls

    # # Victory spin
    # while True:
    #     yield turn_action


def wall_follower_robot(follow: Literal[RelativeDirection.LEFT, RelativeDirection.RIGHT]) -> Algorithm:
    """A robot that follows the wall.

    Returns:
        Robot: The robot's brain.
    """
    def _inner(maze: Maze, goals: set[tuple[int, int]]) -> Robot:
        return _wall_follower_robot(maze, goals, follow=follow)

    return _inner


class SimulationStatus(Enum):
    """TODO"""

    READY = auto()
    IN_PROGRESS = auto()
    FINISHED = auto()
    ERROR = auto()


class Simulator:
    """A micromouse simulator."""
    _status: SimulationStatus
    _robot_pos: tuple[int, int, Direction]

    def __init__(self, alg: Algorithm, maze: Maze, begin: tuple[int, int, Direction], *end: tuple[int, int]):
        self._maze = maze
        self._begin = begin
        self._end = set(end)
        self.restart(alg)

    def restart(self, alg: Algorithm):
        """Restart the simulator."""
        self._robot_maze = Maze.empty(*self._maze.size)
        self._robot_pos = self._begin

        self._robot = alg(self._robot_maze, self._end)

        self._status = SimulationStatus.ERROR
        try:
            state = next(self._robot)
        except StopIteration as stop:
            raise RuntimeError("robot failed to start - stopped before yielding any action") from stop
        except Exception as err:
            raise RuntimeError("robot failed to start - encountered an error") from err
        if state is not Action.READY:
            raise RuntimeError(f"robot malfunction - yielded {state} instead of {Action.READY}")
        self._status = SimulationStatus.READY

    def step(self) -> SimulationStatus:
        """Perform a single robot action."""
        # if self._status not in (SimulationStatus.IN_PROGRESS, SimulationStatus.READY):
        #     # Simulation is not running.
        #     return self._status
        # self._status = SimulationStatus.IN_PROGRESS

        if self._status is SimulationStatus.ERROR:
            return self._status

        if self._status is SimulationStatus.READY:
            self._status = SimulationStatus.IN_PROGRESS

        row, col, facing = self._robot_pos

        try:
            action = self._robot.send(RobotState(*self._robot_pos, self._maze[row, col]))
        except StopIteration:
            self._status = SimulationStatus.FINISHED if (row, col) in self._end else SimulationStatus.ERROR
        except Exception as err:
            self._status = SimulationStatus.ERROR
            raise RuntimeError("robot encountered an error") from err

        match action:
            case Action.READY:
                self._status = SimulationStatus.ERROR
                raise RuntimeError(f"robot malfunction - yielded {action} instead of moving")
            case Action.FORWARD | Action.BACKWARDS:
                if not self._robot_step(facing if action is Action.FORWARD else facing.turn_back()):
                    self._status = SimulationStatus.ERROR
            case Action.TURN_LEFT:
                self._robot_pos = (row, col, facing.turn_left())
            case Action.TURN_RIGHT:
                self._robot_pos = (row, col, facing.turn_right())

        return self._status

    def _robot_step(self, direction: Direction) -> bool:
        """Try to advance the robot in the given direction, return False if not possible."""
        row, col, facing = self._robot_pos

        if direction_to_wall(direction) in self._maze[row, col]:
            # Robot crashed into a wall
            return False

        # No need to check boundaries because our maze is enclosed by walls
        match direction:
            case Direction.NORTH: self._robot_pos = (row - 1, col, facing)
            case Direction.EAST: self._robot_pos = (row, col + 1, facing)
            case Direction.SOUTH: self._robot_pos = (row + 1, col, facing)
            case Direction.WEST: self._robot_pos = (row, col - 1, facing)
            case _: raise AssertionError(f"only the primary directions are supported right now (not {direction})")
        return True

    @property
    def maze(self) -> Maze:  # TODO: readonly version
        """The maze used for in the simulator."""
        return self._maze

    @property
    def begin(self) -> tuple[int, int, Direction]:
        """The starting position in the maze."""
        return self._begin

    @property
    def end(self) -> set[tuple[int, int]]:  # TODO: maybe a readonly view
        """The goal position(s) in the maze."""
        return self._end

    @property
    def robot_maze(self) -> Maze:  # TODO: readonly version
        """The maze that the robot sees."""
        return self._robot_maze

    @property
    def robot_pos(self) -> tuple[int, int, Direction]:
        """The robot's current position."""
        return self._robot_pos

    @property
    def status(self) -> SimulationStatus:
        """The status of the simulator."""
        return self._status