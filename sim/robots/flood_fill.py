"""flood fill robot

Uses flood-fill to get to the goal and then back to the start,
then uses an "optimal" route (by default, also using flood-fill).
"""

from __future__ import annotations

import math

from functools import partial, reduce
from operator import or_
from typing import Protocol, TypedDict, TYPE_CHECKING

from .utils import Action, adjacent_cells, direction_to_cell, shuffled, walls_to_directions
from .utils import build_weighted_graph, dijkstra, identity, mark_unreachable_groups
from .const import predetermined_path_robot
from ..directions import Direction, RelativeDirection
from ..maze import Walls
from ..unionfind import UnionFind

if TYPE_CHECKING:
    from collections.abc import Iterable, Set
    from typing import Callable, Unpack
    from ..maze import ExtendedMaze, ExtraCellInfo
    from .utils import Algorithm, Robot, RobotState, SolverAlgorithm

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


def weight_with_avoid_cells(weight: WeightCalc, avoid: Set[tuple[int, int]], avoid_weight: float = UNREACHABLE_WEIGHT) -> WeightCalc:
    """
    Create a weight generator that avoids certain cells by adding a penalty to their weight.

    Args:
        weight (WeightCalc): A weight function for cells that are not avoided.
        avoid (Set[tuple[int, int]]): The cells to avoid.
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
        goals: Set[tuple[int, int]],
        weight: WeightCalc = simple_flood_weight,
        *,
        force: bool = False,
) -> None:
    """Calculates flood-fill weights *in place*.

    Args:
        maze (ExtendedMaze): The maze to flood.
        goals (Set[tuple[int, int]]): The goals to flood to.
        robot_pos (tuple[int, int]): The starting position for the flood.
        robot_direction (Direction): The starting direction for the flood.
        weight (WeightCalc, optional): The weight function for the flood. Defaults to simple_flood_weight.
        force (bool, optional): Force recalculation, even if the maze didn't change. Default False.
    """
    # Check ``force`` later to update the change status of the maze
    # Check for missing cells last because it is the worst complexity
    if not maze.changed() and not force and all(info.weight is not None for _, _, info in maze.iter_info()):
        return

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
            unreachable = {(r, c) for r, c, _ in maze} - seen
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
        goals: Set[tuple[int, int]],
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

    def _calc_flood_fill(force: bool = False) -> None:
        calc_flood_fill(
            maze=maze,
            goals=goals,
            weight=weight,
            force=force,
        )

    if recalculate_flood:
        _loop_calc_flood_fill = _calc_flood_fill
    else:
        _loop_calc_flood_fill = _do_nothing

    def _assert_turn(r: int, c: int, heading: Direction):
        assert (r, c) == pos, "moved while turning"
        assert maze[r, c] == walls, "walls changed while turning"
        assert heading == new_direction, "turning failed"

    _calc_flood_fill(force=True)

    pos_row, pos_col, heading = yield Action.READY
    maze.route.append(pos := (pos_row, pos_col))

    while pos not in goals:
        maze.extra_info[pos_row, pos_col].visit_cell()
        _loop_calc_flood_fill()
        if math.isinf(maze.extra_info[pos_row, pos_col].weight or 0):
            print("floodmouse: cannot reach goals, giving up")
            break
        walls = maze[pos_row, pos_col]
        print(f"floodmouse: at {pos} facing {heading} with {walls}")
        new_direction = min(
            minor_priority(walls_to_directions(walls)),  # regular / reversed / shuffled
            key=_priority,
        )
        print(f"floodmouse: chose to flood {new_direction}")
        if new_direction == heading:
            print("floodmouse: will move forward")
            action = Action.FORWARD
        elif new_direction == heading.turn_back():
            print("floodmouse: will move in reverse")
            action = Action.BACKWARDS
        else:
            if new_direction == heading.turn_left():
                print("floodmouse: turning left")
                turn_action = Action.TURN_LEFT
            elif new_direction == heading.turn_right():
                print("floodmouse: turning right")
                turn_action = Action.TURN_RIGHT
            else:
                raise AssertionError(f"invalid turn from {heading} to {new_direction}")
            _assert_turn(*(yield turn_action))
            print(f"floodmouse: now facing {heading}, will move forward")
            action = Action.FORWARD
        pos_row, pos_col, heading = yield action
        maze.route.append(pos := (pos_row, pos_col))


def flood_fill_explore(
        maze: ExtendedMaze,
        goals: Set[tuple[int, int]],
        *,
        weight: WeightCalc = simple_flood_weight,
        minor_priority: MinorPriority = shuffled,
) -> Robot:
    """Explore the maze using the flood-fill algorithm.

    Args:
        maze (ExtendedMaze): The maze.
        goals (Set[tuple[int, int]]): The goal cells.
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
    maze.mark_changed()
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

    def _flood_fill_robot_impl(maze: ExtendedMaze, goals: Set[tuple[int, int]]) -> Robot:
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


def simple_flood_fill(maze: ExtendedMaze, goals: Set[tuple[int, int]]) -> Robot:
    """A robot that solves the maze using a simple implementation of the flood-fill algorithm.

    Returns:
        Robot: The robot's brain.
    """
    return flood_fill_robot(final_unknown_penalty=0)(maze, goals)


def dijkstra_solver(
        maze: ExtendedMaze,
        goals: Set[tuple[int, int]],
        *,
        pos: RobotState | None = None,
        unknown_cells: Set[tuple[int, int]] = frozenset(),
) -> Robot:
    """A robot that solves the maze using dijkstra.

    THIS IS A SECOND STEP ROBOT!

    Returns:
        Robot: The robot's brain.
    """
    forward_weight = 1  # basic unit
    reverse_weight = 2  # reverse is twice as slow as full speed ahead
    turn_weight = 1 + 2 + 1  # deceleration penalty + turn time + acceleration penalty

    if pos is None:
        # Remember unknown cells so we can avoid them later
        unknown_cells = {
            (row, col)
            for row, col, info in maze.iter_info()
            if info.visited == 0
        }

        # Reset colors, route and orientation (but keep walls, we should be at the starting position)
        maze.reset_info()
        del maze.route
        pos = yield Action.RESET

    shortest_routes = dijkstra(
        build_weighted_graph(
            maze,
            {
                RelativeDirection.FRONT: forward_weight,
                RelativeDirection.BACK: reverse_weight,
                RelativeDirection.LEFT: turn_weight,
                RelativeDirection.RIGHT: turn_weight,
            },
            start=pos,
            without=unknown_cells,
        ),
        pos[:-1],
    )

    for row, col, info in maze.iter_info():
        info.weight = shortest_routes.get((row, col), (math.inf, None))[0]
        info.color = 'red' if (row, col) in unknown_cells else None

    weight, best = min(  # there is at least 1 route
        (shortest_routes.get(goal, (math.inf, [])) for goal in goals),
        key=lambda weight_route: (weight_route[0], len(weight_route[1])),
    )

    assert math.isfinite(weight)
    assert best[0] == pos[:-1]
    assert best[-1] in goals

    # Do the actual fast route
    yield from predetermined_path_robot(
        maze,
        goals,
        path=best,
        initial_heading=pos[-1],
    )


def _two_step_robot(
        maze: ExtendedMaze,
        goals: Set[tuple[int, int]],
        *,
        explorer: Algorithm = flood_fill_explore,
        solver: SolverAlgorithm = dijkstra_solver,
) -> Robot:
    """Combines 2 robots: one for exploration and another for finding an optimal path.

    Returns:
        Robot: The robot's brain.
    """
    yield from explorer(maze, goals)

    # Remember unknown cells so we can avoid them later
    unknown_cells = {
        (row, col)
        for row, col, info in maze.iter_info()
        if info.visited == 0
    }

    # Reset colors, route and orientation (but keep walls, we should be at the starting position)
    maze.reset_info()
    del maze.route
    pos = yield Action.RESET

    yield from solver(maze, goals, pos=pos, unknown_cells=unknown_cells)


def basic_weighted_flood_fill(maze: ExtendedMaze, goals: Set[tuple[int, int]]) -> Robot:
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


DEFAULT_DIJKSTRA_WEIGHTS: dict[RelativeDirection, float] = {
    RelativeDirection.FRONT: 1,
    RelativeDirection.BACK: 2,
    RelativeDirection.LEFT: 4,
    RelativeDirection.RIGHT: 4,
}


def _calc_unknown_groups(
        maze: ExtendedMaze,
        unknown_color: tuple[int, int, int] | str | None = None,
) -> tuple[UnionFind[tuple[int, int]], set[tuple[int, int]]]:
    groups = UnionFind()
    for row, col, walls, info in maze.iter_all():
        if info.visited > 0:
            if info.color == unknown_color:
                info.color = None
            continue
        called_union = False
        for missing in ~walls:
            match missing:
                case Walls.NORTH:
                    adjacent = (row - 1, col)
                case Walls.EAST:
                    adjacent = (row, col + 1)
                case Walls.SOUTH:
                    adjacent = (row + 1, col)
                case Walls.WEST:
                    adjacent = (row, col - 1)
            if maze.extra_info[adjacent].visited == 0:
                groups.union((row, col), adjacent)
                called_union = True
        if called_union:
            if info.color is None and unknown_color is not None:
                info.color = unknown_color
        else:
            info.color = None
            info.visited = 1
    return groups, reduce(or_, groups.iter_sets(), set())


def flood_fill_thourough_explorer(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        maze: ExtendedMaze,
        goals: Set[tuple[int, int]],
        *,
        percentage: float = 1.0,
        flood_weight: WeightCalc = simple_flood_weight,
        minor_priority: MinorPriority = identity,
) -> Robot:
    """A robot that explores the maze using the flood-fill algorithm.

    THIS IS MEANT AS A FIRST STEP ROBOT!

    Args:
        maze (ExtendedMaze): The maze.
        goals (Set[tuple[int, int]]): The final goals.
        percentage (float, optional): The percentage of the maze to explore before returning home. Defaults to 1.0.
        flood_weight (WeightCalc, optional): Exploration weight. Defaults to simple_flood_weight.
        minor_priority (MinorPriority, optional): Exploration minor priority. Defaults to identity.

    Returns:
        Robot: The robot's brain.
    """
    if not 0 < percentage <= 1.0:
        raise ValueError(f"invalid percentage: {percentage!r}")

    unknown_color = 'blue'

    # First, find the goal
    # We cannot use ``yield from`` because we need the final reply from the yield
    print(f"flood hunter: looking for {goals=}")
    flood_bot = single_flood_fill(
        maze,
        goals,
        weight=flood_weight,
        minor_priority=minor_priority,
    )
    assert next(flood_bot, None) is Action.READY
    pos = yield Action.READY
    start = pos[:-1]
    while True:
        try:
            pos = yield flood_bot.send(pos)
        except StopIteration:
            break

    print(f"flood hunter: found goals! {start=}, {maze.explored_cells_percentage()=:.02%}/{percentage=:.02%}")
    while maze.explored_cells_percentage() < percentage:
        print(f"flood hunter: pos={tuple(pos)}, {maze.explored_cells_percentage()=:.02%}/{percentage=:.02%}")
        # print(_render_maze(maze, goals=goals, pos=pos))
        mark_unreachable_groups(maze, pos[:-1])

        unknown, all_unknown = _calc_unknown_groups(maze, unknown_color)

        routes = dijkstra(
            build_weighted_graph(maze, DEFAULT_DIJKSTRA_WEIGHTS, start=pos),
            pos[:-1],
            goals=all_unknown,
        )

        potential_routes = {
            # The start gets 'inf' so that is is chosen last (reaching the start ends the exploration)
            best[1]: (best[0] / len(group)) if start not in group else math.inf
            for group in unknown
            if math.isfinite((best := max((routes.get(cell, (math.inf, []))[0], cell) for cell in group))[0])
        }

        # print(f"flood hunter: {potential_routes=}")

        if not potential_routes:
            # Nothing to do
            print(f"flood hunter: {routes=}")
            break

        dest = min(potential_routes, key=potential_routes.__getitem__)
        maze.extra_info[dest].color = 'green'
        print(f"flood hunter: hunting {dest}")

        flood_bot = single_flood_fill(
            maze,
            {dest},
            weight=simple_flood_weight_with_norm_visit_bias,
            minor_priority=minor_priority,
        )
        assert next(flood_bot, None) is Action.READY
        while True:
            # print(maze.render_extra(goals=goals, pos=pos))
            if math.isinf(maze.extra_info[pos[:-1]].weight):
                print("flood hunter: dest is unreachable")
                break
            if dest not in all_unknown:
                print("flood hunter: dest discovered without visiting")
                break
            try:
                pos = yield flood_bot.send(pos)
            except StopIteration:
                print(f"flood hunter: reached dest - {pos=}")
                break
            _, all_unknown = _calc_unknown_groups(maze, unknown_color)
        maze.extra_info[dest].color = None

    print(f"flood hunter: done exploring - {maze.explored_cells_percentage()=:.02%}/{percentage=:.02%}")
    while pos[:-1] != start:
        routes = dijkstra(
            build_weighted_graph(maze, DEFAULT_DIJKSTRA_WEIGHTS, start=pos),
            pos[:-1],
            goals={start},
        )
        print(f"flood hunter: @{pos} -> {routes[start][1]}")
        assert routes[start][1][0] == pos[:-1], f"robot is at {pos[:-1]} but route starts at {routes[start][1][0]}"
        return_bot = predetermined_path_robot(
            maze,
            {start},
            path=routes[start][1],
            initial_heading=pos[-1],
        )
        assert next(return_bot, None) is Action.READY
        pos = yield Action.READY
        while True:
            try:
                pos = yield return_bot.send(pos)
            except StopIteration:
                assert pos[:-1] == start
                break
            if maze.changed():
                print("flood hunter: encountered a wall while going home")
                maze.mark_changed()  # restore mark
                break  # recalculate route, a new wall was added


def thourough_flood_fill(maze: ExtendedMaze, goals: Set[tuple[int, int]]) -> Robot:
    """A robot that solves the maze using an advanced implementation of the flood-fill algorithm and some dijkstra.

    Returns:
        Robot: The robot's brain.
    """
    return _two_step_robot(
        maze,
        goals,
        explorer=partial(
            flood_fill_thourough_explorer,
            flood_weight=simple_flood_weight_with_strong_visit_bias,
            minor_priority=identity,
        ),
    )
