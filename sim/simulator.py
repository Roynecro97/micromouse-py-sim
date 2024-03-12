"""Simulator
"""
from __future__ import annotations

import sys
import time

from contextlib import contextmanager
from enum import auto, Enum
from typing import TYPE_CHECKING

from .maze import Direction, ExtendedMaze, Maze, RelativeDirection
from .robots import Action, RobotState
from .robots.utils import build_weighted_graph, dijkstra, direction_to_wall

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .robots import Algorithm, Robot
    from .maze import ExtraCellInfo


class SimulationStatus(Enum):
    """TODO"""

    READY = auto()
    IN_PROGRESS = auto()
    IN_PROGRESS_FOUND_DEST = auto()
    FINISHED = auto()
    ERROR = auto()


@contextmanager
def timed(title: str = ""):
    """A contextmanager for measuring time (in seconds)."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        t1 = time.perf_counter()
        elapsed = t1 - t0
        if exc := sys.exception():
            result = 'failed'
            tail = f" ({type(exc).__name__}: {exc!s})"
        else:
            result = 'completed'
            tail = ""
        print(f"{title}{title and ' '}{result} in {elapsed:0.6f}s{tail}")


class Simulator:  # pylint: disable=too-many-instance-attributes
    """A micromouse simulator."""
    _status: SimulationStatus
    __robot_pos: tuple[int, int, Direction]

    def __init__(self, alg: Algorithm, maze: Maze, begin: tuple[int, int, Direction], end: Iterable[tuple[int, int]]):
        self._maze = ExtendedMaze.full_from_maze(maze)
        self._begin = begin
        self._end = set(end)

        if self._begin[:-1] not in self._end and direction_to_wall(self._begin[-1]) in self._maze[self._begin[:-1]]:
            raise ValueError("robot starts facing a wall")
        if not self._end:
            raise ValueError("must specify at least 1 end cell")
        if not self.connected(self._begin[:-1], self._end):
            raise ValueError("the starting position (begin) is not connected to all goal positions (end)")

        self._maze.route = min(
            dijkstra(
                build_weighted_graph(
                    self._maze,
                    {
                        RelativeDirection.FRONT: 1,
                        RelativeDirection.BACK: 2,
                        RelativeDirection.LEFT: 4,
                        RelativeDirection.RIGHT: 4,
                    },
                    start=self._begin,
                ),
                self._begin[:-1],
                goals=self._end,
            ).values(),
        )[1]

        self.restart(alg)

    def restart(self, alg: Algorithm):
        """Restart the simulator."""
        print(f"sim: restarting with a {self.maze.height}x{self.maze.width} maze")
        print(f"sim: robot will start at {self._begin[:-1]} facing {self._begin[-1]}")
        self._maze.reset_info()
        self._robot_maze = ExtendedMaze.empty(self._maze.height, self.maze.width)
        self._robot_pos = self._begin

        self._robot = alg(self._robot_maze, self._end)

        self._status = SimulationStatus.ERROR
        try:
            with timed("sim: robot init"):
                state = next(self._robot)
        except StopIteration as stop:
            raise RuntimeError("robot failed to start - stopped before yielding any action") from stop
        except Exception as err:
            raise RuntimeError("robot failed to start - encountered an error") from err
        if state is not Action.READY:
            raise RuntimeError(f"robot malfunction - yielded {state} instead of {Action.READY}")
        self._status = SimulationStatus.READY
        print("sim: robot is ready")

    def step(self) -> SimulationStatus:
        """Perform a single robot action."""
        # if self._status not in (SimulationStatus.IN_PROGRESS, SimulationStatus.READY):
        #     # Simulation is not running.
        #     return self._status
        # self._status = SimulationStatus.IN_PROGRESS

        if self._status is SimulationStatus.ERROR:
            print(f"sim: refusing to step, status is {self.status}")
            return self._status

        if self._status is SimulationStatus.READY:
            print("sim: starting progress")
            self._status = SimulationStatus.IN_PROGRESS

        row, col, facing = self._robot_pos
        print(f"sim: robot is at {(row, col)} facing {facing}")

        try:
            with timed("sim: robot step"):
                action = self._robot.send(RobotState(*self._robot_pos))
        except StopIteration:
            self._status = SimulationStatus.FINISHED if (row, col) in self._end else SimulationStatus.ERROR
            print(f"sim: robot stopped, status is {self.status}")
            return self._status
        except Exception as err:
            self._status = SimulationStatus.ERROR
            raise RuntimeError("robot encountered an error") from err

        print(f"sim: selected action is {action}")

        match action:
            case Action.READY:
                pass
                # self._status = SimulationStatus.ERROR
                # raise RuntimeError(f"robot malfunction - yielded {action} instead of moving")
            case Action.RESET:
                print("sim: robot asked for reset")
                if self._robot_pos[:-1] != self._begin[:-1] or self._status is not SimulationStatus.IN_PROGRESS_FOUND_DEST:
                    raise RuntimeError("reset must be done from the starting position and after finding the goal")
                self._maze.reset_info()
                self._robot_pos = self._begin
                self._status = SimulationStatus.READY  # Allow the robot to call ready again
            case Action.FORWARD | Action.BACKWARDS:
                if not self._robot_step(facing if action is Action.FORWARD else facing.turn_back()):
                    print("sim: step error")
                    self._status = SimulationStatus.ERROR
                elif self._robot_pos[:-1] in self._end:
                    self._status = SimulationStatus.IN_PROGRESS_FOUND_DEST
            case Action.TURN_LEFT:
                self._robot_pos = (row, col, facing.turn_left())
                print("sim: robot turned left")
            case Action.TURN_RIGHT:
                self._robot_pos = (row, col, facing.turn_right())
                print("sim: robot turned right")

        return self._status

    def _robot_step(self, direction: Direction) -> bool:
        """Try to advance the robot in the given direction, return False if not possible."""
        row, col, facing = self._robot_pos

        print(f"sim: stepping from {(row, col)} to {direction} (while facing {facing})")

        if direction_to_wall(direction) in self._maze[row, col]:
            print(f"sim: crashed! {direction_to_wall(direction)=!s} to {self._maze[row, col]=!s}")
            # Robot crashed into a wall
            return False

        # No need to check boundaries because our maze is enclosed by walls
        match direction:
            case Direction.NORTH: self._robot_pos = (row - 1, col, facing)
            case Direction.EAST: self._robot_pos = (row, col + 1, facing)
            case Direction.SOUTH: self._robot_pos = (row + 1, col, facing)
            case Direction.WEST: self._robot_pos = (row, col - 1, facing)
            case _: raise AssertionError(f"only the primary directions are supported right now (not {direction})")

        print(f"sim: robot is now at {self._robot_pos[:-1]} facing {self._robot_pos[-1]}")
        return True

    @property
    def _robot_pos(self) -> tuple[int, int, Direction]:
        return self.__robot_pos

    @_robot_pos.setter
    def _robot_pos(self, new_pos: tuple[int, int, Direction]):
        self.__robot_pos = new_pos

        robot_pos = self.__robot_pos[:-1]
        self._robot_maze[robot_pos] = self._maze[robot_pos]
        self._robot_maze.extra_info[robot_pos].visit_cell()
        info: ExtraCellInfo = self._maze.extra_info[robot_pos]
        info.visit_cell()

    @property
    def maze(self) -> ExtendedMaze:  # TODO: readonly version
        """The maze used in the simulator."""
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
    def robot_maze(self) -> ExtendedMaze:  # TODO: readonly version
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

    def connected(self, a: tuple[int, int] | Iterable[tuple[int, int]], b: tuple[int, int] | Iterable[tuple[int, int]]) -> bool:
        """Check if two cells (or cell groups) are connected in the maze."""
        # Calculate connectivity
        connectivity = self._maze.connectivity

        # Normalize input to 2 sets
        def _normalize(maybe_group: tuple[int, int] | Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
            if not isinstance(maybe_group, tuple):
                return set(maybe_group)
            if len(maybe_group) != 2 or isinstance(maybe_group[0], tuple):
                return set(maybe_group)  # type: ignore
            return {maybe_group}  # type: ignore

        # Check connectivity
        return all(all(connectivity.connected(point_a, point_b) for point_b in _normalize(b)) for point_a in _normalize(a))
