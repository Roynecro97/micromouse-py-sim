"""flood fill robot

Uses flood-fill to get to the goal and then back to the start,
then uses an "optimal" route (by default, also using flood-fill).
"""

from __future__ import annotations

import math

from functools import partial
from typing import Protocol, TypedDict, TYPE_CHECKING

from .utils import Action, adjacent_cells, direction_to_cell, shuffled, walls_to_directions
from .utils import build_weighted_graph, dijkstra, identity
from .const import predetermined_path_robot
from ..maze import Direction, RelativeDirection

if TYPE_CHECKING:
    from collections.abc import Iterable, Set
    from typing import Callable, Unpack
    from ..maze import ExtendedMaze, ExtraCellInfo
    from .utils import Algorithm, Robot

    type MinorPriority = Callable[[list[Direction]], Iterable[Direction]]


class WeightArgs(TypedDict, total=True):
    """Arguments for weighted calculators."""
    maze: ExtendedMaze
    cell: tuple[int, int]
    info: ExtraCellInfo
    marker: int


class WeightCalc(Protocol):  # pylint: disable=too-few-public-methods
    """A flood-fill weight calculator."""

    def __call__(self, **kwargs: Unpack[WeightArgs]) -> float:
        ...


UNREACHABLE_WEIGHT = math.inf


def simple_flood_weight(**kwargs: Unpack[WeightArgs]) -> float:
    """Simple length-based flood fill weight (aka weight-less flood fill)."""
    return kwargs['marker']


def simple_flood_weight_with_norm_visit_bias(**kwargs: Unpack[WeightArgs]) -> float:
    """
    Simple length-based flood fill weight (aka weight-less flood fill) but with
    a weak bias against seen cells.
    """
    if kwargs['marker'] == 0:
        return 0
    return kwargs['marker'] + bool(kwargs['info'].visited)


def simple_flood_weight_with_strong_visit_bias(**kwargs: Unpack[WeightArgs]) -> float:
    """
    Simple length-based flood fill weight (aka weight-less flood fill) but with
    a weak bias against seen cells.
    """
    if kwargs['marker'] == 0:
        return 0
    return kwargs['marker'] + round(kwargs['info'].visited * 0.5)


def weight_with_avoid_cells(weight: WeightCalc, avoid: set[tuple[int, int]], avoid_weight: float = UNREACHABLE_WEIGHT) -> WeightCalc:
    """
    Create a weight generator that avoids certain cells by adding a penalty to their weight.

    Args:
        weight (WeightCalc): A weight function for cells that are not avoided.
        avoid (set[tuple[int, int]]): The cells to avoid.
        avoid_weight (float, optional):
            The weight penalty for avoided cells, this is added to the regular weight.
            Defaults to UNREACHABLE_WEIGHT.

    Returns:
        WeightCalc: A weight calculator based on the provided calculator that avoids the specified cells.
    """
    def weight_and_avoid(**kwargs: Unpack[WeightArgs]) -> float:
        """Calc weights, avoiding some cells.

        Returns:
            float: Weight.
        """
        penalty = avoid_weight if kwargs['cell'] in avoid else 0
        return weight(**kwargs) + penalty

    return weight_and_avoid


def calc_flood_fill(
        maze: ExtendedMaze,
        goals: set[tuple[int, int]],
        weight: WeightCalc = simple_flood_weight,
) -> None:
    """Calculates flood-fill weights *in place*.

    Args:
        maze (ExtendedMaze): The maze to flood.
        goals (set[tuple[int, int]]): The goals to flood to.
        robot_pos (tuple[int, int]): The starting position for the flood.
        robot_direction (Direction): The starting direction for the flood.
        weight (WeightCalc, optional): The weight function for the flood. Defaults to simple_flood_weight.
    """
    def _calc_weight(**kwargs: Unpack[WeightArgs]) -> float:
        try:
            return weight(**kwargs)
        except KeyError as kerr:
            if len(kerr.args) == 1 and kerr.args[0] in WeightArgs.__annotations__:
                raise TypeError(f"missing required keyword argument: {kerr.args[0]!r}") from kerr
            raise

    seen = set()
    current = goals
    marker = 0
    while True:
        for cell in current:
            info: ExtraCellInfo = maze.extra_info[cell]
            info.weight = _calc_weight(
                maze=maze,
                cell=cell,
                info=info,
                marker=marker,
            )
        marker += 1
        seen.update(current)
        current = adjacent_cells(maze, current, seen)
        if len(seen) >= maze.cell_size:
            break
        if not current:
            unreachable = {(r, c) for r in range(maze.height) for c in range(maze.width)} - seen
            print("unreachable:", unreachable)
            for cell in unreachable:
                maze.extra_info[cell].weight = UNREACHABLE_WEIGHT
            seen.update(unreachable)
            break  # unreachable cells detected
    assert len(seen) == maze.cell_size, f"new cells created ({len(seen)}/{maze.cell_size})"


def _do_nothing() -> None:
    pass


def single_flood_fill(  # pylint: disable=too-many-locals
        maze: ExtendedMaze,
        goals: set[tuple[int, int]],
        *,
        weight: WeightCalc = simple_flood_weight,
        minor_priority: MinorPriority = shuffled,
        recalculate_flood: bool = True,
) -> Robot:
    """A robot that solves the maze using a simple implementation of the flood-fill algorithm.

    Returns:
        Robot: The robot's brain.
    """
    def _priority(direction: Direction) -> tuple[float, int]:
        cell: ExtraCellInfo = maze.extra_info[dst := direction_to_cell(pos, direction)]
        assert cell.weight is not None, f"unweighted cell: {dst}"
        return cell.weight, cell.visited

    def _calc_flood_fill() -> None:
        calc_flood_fill(
            maze=maze,
            goals=goals,
            weight=weight,
        )

    if recalculate_flood:
        _initial_calc_flood_fill = _do_nothing
        _loop_calc_flood_fill = _calc_flood_fill
    else:
        _initial_calc_flood_fill = _calc_flood_fill
        _loop_calc_flood_fill = _do_nothing

    def _assert_turn(r: int, c: int, facing: Direction):
        assert (r, c) == pos, "moved while turning"
        assert maze[r, c] == walls, "walls changed while turning"
        assert facing == new_direction, "turning failed"

    pos_row, pos_col, facing = yield Action.READY
    maze.route.append(pos := (pos_row, pos_col))

    _initial_calc_flood_fill()

    while pos not in goals:
        maze.extra_info[pos_row, pos_col].visit_cell()
        _loop_calc_flood_fill()
        walls = maze[pos_row, pos_col]
        print(f"floodmouse: at {pos} facing {facing} with {walls}")
        new_direction = min(
            minor_priority(walls_to_directions(walls)),  # regular / reversed / shuffled
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
            _assert_turn(*(yield turn_action))
            print(f"floodmouse: now facing {facing}, will move forward")
            action = Action.FORWARD
        pos_row, pos_col, facing = yield action
        maze.route.append(pos := (pos_row, pos_col))


def flood_fill_explore(
        maze: ExtendedMaze,
        goals: set[tuple[int, int]],
        *,
        weight: WeightCalc = simple_flood_weight,
        minor_priority: MinorPriority = shuffled,
) -> Robot:
    """Explore the maze using the flood-fill algorithm.

    Args:
        maze (ExtendedMaze): The maze.
        goals (set[tuple[int, int]]): The goal cells.
        weight (WeightCalc, optional): The weight function. Defaults to simple_flood_weight.
        minor_priority (MinorPriority, optional):
            The last priority in selecting cells. Defaults to shuffled.

    Returns:
        Robot: The robot's exploration brain.
    """
    # Initial flood-fill to the goal
    yield from single_flood_fill(
        maze,
        goals,
        weight=weight,
        minor_priority=minor_priority,
    )

    # Mark the starting point as the new goal and flood-fill to get there
    starting_pos = maze.route[0]
    maze.extra_info[starting_pos].color = "green"
    for goal in goals:
        maze.extra_info[goal].color = "blue"
    yield from single_flood_fill(
        maze,
        {starting_pos},
        weight=weight,
        minor_priority=minor_priority,
    )


def flood_fill_robot(  # pylint: disable=too-many-arguments
        *,
        weight: WeightCalc = simple_flood_weight,
        minor_priority: MinorPriority = shuffled,
        final_weight: WeightCalc | None = None,
        final_minor_priority: MinorPriority | None = None,
        final_unknown_penalty: float = UNREACHABLE_WEIGHT,
) -> Algorithm:
    """A robot that solves the maze with only the flood

    Args:
        weight (WeightCalc, optional):
            The weight function for exploration. Defaults to simple_flood_weight.
        minor_priority (MinorPriority, optional):
            The priority tie-breaker for exploration. Defaults to shuffled.
        final_weight (WeightCalc | None, optional):
            The weight function for the final flood fill, if ``None``, ``weight`` is used. Defaults to None.
        final_minor_priority (MinorPriority | None, optional):
            The priority tie-breaker for the final flood fill, if ``None``, ``weight`` is used. Defaults to None.
        final_unknown_penalty (bool, optional):
            Weight penalty for unknown cells in the final flood fill run.
            The final run should be the optimal run so unknown cells should usually be avoided.
            Defaults to UNREACHABLE_WEIGHT which means "avoid at all costs".

    Returns:
        Robot: The robot's brain
    """
    if final_weight is None:
        final_weight = weight
    if final_minor_priority is None:
        final_minor_priority = minor_priority

    def _flood_fill_robot_impl(maze: ExtendedMaze, goals: set[tuple[int, int]]) -> Robot:
        """A robot that solves the maze using a simple implementation of the flood-fill algorithm.

        Returns:
            Robot: The robot's brain.
        """
        yield from flood_fill_explore(
            maze,
            goals,
            weight=weight,
            minor_priority=minor_priority,
        )

        # Remember unknown cells so we can avoid them later
        unknown_cells = {
            (row, col)
            for row, col, info in maze.iter_info()
            if info.visited == 0
        }
        # Reset colors, route and orientation (but keep walls, we should be at the starting position)
        maze.reset_info()
        del maze.route
        yield Action.RESET
        # Do the actual fast route
        yield from single_flood_fill(
            maze,
            goals,
            weight=weight_with_avoid_cells(final_weight, unknown_cells, final_unknown_penalty),
            minor_priority=final_minor_priority,
            recalculate_flood=math.isinf(final_unknown_penalty) and final_unknown_penalty > 0,
        )

    return _flood_fill_robot_impl


def simple_flood_fill(maze: ExtendedMaze, goals: set[tuple[int, int]]) -> Robot:
    """A robot that solves the maze using a simple implementation of the flood-fill algorithm.

    Returns:
        Robot: The robot's brain.
    """
    return flood_fill_robot(final_unknown_penalty=0)(maze, goals)


def dijkstra_solver(maze: ExtendedMaze, goals: set[tuple[int, int]]) -> Robot:
    """A robot that solves the maze using dijkstra.

    THIS IS A SECOND STEP ROBOT!

    Returns:
        Robot: The robot's brain.
    """
    forward_weight = 1  # basic unit
    reverse_weight = 2  # reverse is twice as slow as full speed ahead
    turn_weight = 1 + 2 + 1  # deceleration penalty + turn time + acceleration penalty

    # Remember unknown cells so we can avoid them later
    unknown_cells = {
        (row, col)
        for row, col, info in maze.iter_info()
        if info.visited == 0
    }

    # Reset colors, route and orientation (but keep walls, we should be at the starting position)
    maze.reset_info()
    del maze.route
    pos_row, pos_col, facing = yield Action.RESET

    shortest_routes = dijkstra(
        build_weighted_graph(
            maze,
            {
                RelativeDirection.FRONT: forward_weight,
                RelativeDirection.BACK: reverse_weight,
                RelativeDirection.LEFT: turn_weight,
                RelativeDirection.RIGHT: turn_weight,
            },
            without=unknown_cells,
        ),
        (pos_row, pos_col, facing),
    )

    for row, col, info in maze.iter_info():
        info.weight = shortest_routes.get((row, col), (math.inf, None))[0]
        info.color = 'red' if (row, col) in unknown_cells else None

    weight, best = min(  # there is at least 1 route
        (shortest_routes.get(goal, (math.inf, [])) for goal in goals),
        key=lambda weight_route: (weight_route[0], len(weight_route[1])),
    )

    assert math.isfinite(weight)
    assert best[0] == (pos_row, pos_col)
    assert best[-1] in goals

    # Do the actual fast route
    yield from predetermined_path_robot(
        maze,
        goals,
        path=best,
        initial_heading=facing,
    )


def _two_step_robot(
        maze: ExtendedMaze,
        goals: set[tuple[int, int]],
        *,
        explorer: Algorithm = flood_fill_explore,
        solver: Algorithm = dijkstra_solver,
) -> Robot:
    """Combines 2 robots: one for exploration and another for finding an optimal path.

    Returns:
        Robot: The robot's brain.
    """
    yield from explorer(maze, goals)
    yield from solver(maze, goals)


def basic_weighted_flood_fill(maze: ExtendedMaze, goals: set[tuple[int, int]]) -> Robot:
    """A robot that solves the maze using a simple implementation of the flood-fill algorithm.

    Returns:
        Robot: The robot's brain.
    """
    return _two_step_robot(
        maze,
        goals,
        explorer=partial(
            flood_fill_explore,
            weight=simple_flood_weight_with_strong_visit_bias,
            minor_priority=identity,
        ),
    )
