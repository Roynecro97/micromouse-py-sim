"""flood fill robot

Uses flood-fill to get to the goal and then back to the start,
then uses an optimal route.
"""

from __future__ import annotations

from typing import Protocol, TypedDict, TYPE_CHECKING

from .utils import Action, adjacent_cells, direction_to_cell, shuffled, walls_to_directions
from ..maze import Direction

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Callable, Unpack
    from ..maze import ExtendedMaze, ExtraCellInfo
    from .utils import Algorithm, Robot

    type MinorPriority = Callable[[list[Direction]], Iterable[Direction]]


class WeightArgs(TypedDict, total=True):
    """Arguments for weighted calculators."""
    maze: ExtendedMaze
    cell: tuple[int, int]
    marker: int
    robot_pos: tuple[int, int]
    robot_direction: Direction


class WeightCalc(Protocol):  # pylint: disable=too-few-public-methods
    """A flood-fill weight calculator."""

    def __call__(self, **kwargs: Unpack[WeightArgs]) -> float:
        ...


UNREACHABLE_WEIGHT = float('inf')



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
    return kwargs['marker'] + bool(kwargs['maze'].extra_info[kwargs['cell']].visited)


def simple_flood_weight_with_strong_visit_bias(**kwargs: Unpack[WeightArgs]) -> float:
    """
    Simple length-based flood fill weight (aka weight-less flood fill) but with
    a weak bias against seen cells.
    """
    if kwargs['marker'] == 0:
        return 0
    return kwargs['marker'] + kwargs['maze'].extra_info[kwargs['cell']].visited


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
        robot_pos: tuple[int, int],
        robot_direction: Direction,
        weight: WeightCalc = simple_flood_weight,
):
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
                marker=marker,
                robot_pos=robot_pos,
                robot_direction=robot_direction,
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


def single_flood_fill(
        maze: ExtendedMaze,
        goals: set[tuple[int, int]],
        *,
        weight: WeightCalc = simple_flood_weight,
        minor_priority: MinorPriority = shuffled,
) -> Robot:
    """A robot that solves the maze using a simple implementation of the flood-fill algorithm.

    Returns:
        Robot: The robot's brain.
    """
    def _priority(direction: Direction) -> tuple[float, int]:
        cell: ExtraCellInfo = maze.extra_info[dst := direction_to_cell(pos, direction)]
        assert cell.weight is not None, f"unweighted cell: {dst}"
        return cell.weight, cell.visited

    pos_row, pos_col, facing = yield Action.READY
    maze.route.append(pos := (pos_row, pos_col))

    while pos not in goals:
        maze.extra_info[pos_row, pos_col].visit_cell()
        calc_flood_fill(
            maze=maze,
            goals=goals,
            robot_pos=(pos_row, pos_col),
            robot_direction=facing,
            weight=weight,
        )
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
            r, c, facing = yield turn_action
            assert (r, c) == pos, "moved while turning"
            assert maze[r, c] == walls, "walls changed while turning"
            assert facing == new_direction, "turning failed"
            print(f"floodmouse: now facing {facing}, will move forward")
            action = Action.FORWARD
        pos_row, pos_col, facing = yield action
        maze.route.append(pos := (pos_row, pos_col))


def _simple_flood_fill(maze: ExtendedMaze, goals: set[tuple[int, int]]) -> Robot:
    """A robot that solves the maze using a simple implementation of the flood-fill algorithm.

    Returns:
        Robot: The robot's brain.
    """
    return single_flood_fill(maze, goals)


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
    yield from single_flood_fill(maze, goals, weight=weight, minor_priority=minor_priority)
    # Mark the starting point as the new goal and flood-fill to get there
    starting_pos = maze.route[0]
    maze.extra_info[starting_pos].color = "green"
    for goal in goals:
        maze.extra_info[goal].color = "blue"
    yield from single_flood_fill(maze, {starting_pos}, weight=weight, minor_priority=minor_priority)


def flood_fill_robot(
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
        yield from flood_fill_explore(maze, goals, weight=weight, minor_priority=minor_priority)

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
        )

    return _flood_fill_robot_impl


def simple_flood_fill(maze: ExtendedMaze, goals: set[tuple[int, int]]) -> Robot:
    """A robot that solves the maze using a simple implementation of the flood-fill algorithm.

    Returns:
        Robot: The robot's brain.
    """
    return flood_fill_robot()(maze, goals)
