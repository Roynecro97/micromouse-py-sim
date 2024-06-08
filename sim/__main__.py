"""main entrypoint for the simulator"""
from __future__ import annotations

import argparse
import sys

from importlib.metadata import entry_points

from .directions import Direction
from .front import (
    direction,
    LoadPreset,
    maze,
    position_set,
    position,
    Renderer,
)
from .maze import Maze
from .robots import idle_robot, load_robots
from .robots.utils import walls_to_directions
from .simulator import Simulator


def _load_gui_engines() -> dict[str, type[Renderer]]:
    engines: dict[str, type[Renderer]] = {}
    for renderer in entry_points(group='micromouse.gui'):
        try:
            renderer_class = renderer.load()
        except (AttributeError, ImportError) as err:
            print(f"warning: failed to load renderer {renderer.name}: {err}")
            continue
        if not issubclass(renderer_class, Renderer):
            print(f"warning: failed to load renderer {renderer.name}: must inherit from {Renderer.__module__}.{Renderer.__qualname__}")
            continue
        if renderer.name in engines:
            print(
                f"warning: {renderer.name} from {renderer.module}:{renderer.attr} overrides "
                f"{engines[renderer.name].__module__}:{engines[renderer.name].__qualname__}"
            )
        engines[renderer.name] = renderer_class
    return engines


def main():
    """
    The main entrypoint for the simulation, loads a simulation based on
    commandline arguments and triggers the selected renderer entrypoint.

    TODO: allow more than just the gui entrypoint.
    TODO: extended argument parser.
    TODO: add utilities like maze template creator/viewer/validator/converter...
    """
    parser = argparse.ArgumentParser(
        description="Micromouse simulator.",
    )
    subparsers = parser.add_subparsers(
        title='actions',
        description="Different action modes.",
        dest='action',
        required=True,
        help="Different action modes ??.",
    )
    sim = subparsers.add_parser(
        'sim',
        description="Run the simulator.",
    )
    sim.add_argument(
        '-m', '--maze',
        type=maze,
        default=Maze.from_maze_text("+---+\n|   |\n+---+\n"),
        help="The maze to load. May be a file or a maze input. (default: 1x1 maze)",
    )
    sim.add_argument(
        '-s', '--start-pos',
        type=position,
        default=(0, 0),
        help="The starting position. (default: the top-left corner - (0, 0))",
    )
    sim.add_argument(
        '-d', '--start-direction',
        type=direction,
        help="The starting direction. (default: a valid direction for the starting position)",
    )
    sim.add_argument(
        '-g', '--goals',
        type=position_set,
        # default={(M - 1, N - 1)},
        help="The goal positions. (default: the bottom-right corner - (M-1, N-1) in a MxN maze)",
    )
    sim.add_argument(
        '-p', '--preset',
        action=LoadPreset,
        dest='maze,start_pos,start_direction,goals',
        presets_file='mazes/presets.json',
        help="A maze+start+goals preset to load.",
    )
    gui_engines = _load_gui_engines()
    sim.add_argument(
        '-e', '--engine',
        choices=gui_engines,
        default='default',
        help="The render engine to use. (default: a simple pygame GUI renderer, requires the 'gui' feature)",
    )

    args = parser.parse_args()
    if args.goals is None:
        args.goals = {(args.maze.height - 1, args.maze.width - 1)}
    if args.start_direction is None:
        args.start_direction = (walls_to_directions(args.maze[args.start_pos]) or [Direction.NORTH])[0]

    load_robots()

    print(args)

    if args.maze:
        # Maze.render = '\n'.join(''.join(row) for row in self.render_screen(charset, cell_width, cell_height, force_corners)) + '\n'
        screen = args.maze.render_screen()
        start_row, start_col = args.start_pos
        match args.start_direction:
            case Direction.NORTH:
                start_mark = '^'
            case Direction.EAST:
                start_mark = '>'
            case Direction.SOUTH:
                start_mark = 'v'
            case Direction.WEST:
                start_mark = '<'
            case _:
                start_mark = 'S'
        screen[start_row * 2 + 1, start_col * 4 + 2] = start_mark
        for (goal_row, goal_col) in args.goals:
            screen[goal_row * 2 + 1, goal_col * 4 + 2] = '@'
        print('\n'.join(''.join(row) for row in screen))
        # print(args.maze.render(), end="")
    else:
        print("no maze")
        sys.exit(1)

    gui_engines[args.engine](Simulator(
        alg=idle_robot,
        maze=args.maze,
        begin=args.start_pos + (args.start_direction,),
        end=args.goals,
    )).run()


if __name__ == '__main__':
    main()
