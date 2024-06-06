"""main entrypoint for the simulator"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

from collections.abc import Set
from importlib.metadata import entry_points
from io import StringIO
from typing import TypedDict, TYPE_CHECKING

from .directions import Direction
from .front import Renderer
from .maze import Maze
from .robots import idle_robot, load_robots
from .robots.utils import walls_to_directions
from .simulator import Simulator

if TYPE_CHECKING:
    from collections.abc import Sequence


def __maze_from_num(num: str) -> Maze:
    return Maze.from_num_file(StringIO(num))


def __maze_from_maz(maz: str) -> Maze:
    if re.fullmatch(r'(?:\\x[a-fA-F0-9]{2}|\s)+', maz):
        maz_bytes = bytes.fromhex(maz.replace('\\x', ''))
    elif re.fullmatch(r'(?:0x[a-fA-F0-9]{1,2}\s*,?|\s|[\[\](){}])+', maz):
        maz_bytes = bytes(int(byte, 16) for byte in re.sub(r'[\[\](){}\s]', '', maz).split(','))
    elif re.fullmatch(r'(?:[a-fA-F0-9]{2}|\s)+', maz):
        maz_bytes = bytearray.fromhex(maz)
    elif all(0 <= ord(c) <= 0xF for c in maz):
        maz_bytes = maz.encode(encoding="ASCII")
    else:
        raise argparse.ArgumentTypeError("invalid maz format")
    return Maze.from_maz(maz_bytes)


def __maze(arg: str) -> Maze:
    """Maze type for argparse."""
    typ = ''
    if ':' in arg:
        typ, sep, arg = arg.partition(':')
        arg = arg.lstrip()
        if not sep and not typ:
            raise argparse.ArgumentTypeError("found ':' but maze type is empty")
    match typ:
        case 'maze' | '':
            file_parser = Maze.from_file
            text_parser = Maze.from_maze_text
        case 'maz':
            file_parser = Maze.from_maz_file
            text_parser = __maze_from_maz
        case 'num':
            file_parser = Maze.from_num_file
            text_parser = __maze_from_num
        case 'csv':
            file_parser = Maze.from_csv_file
            text_parser = Maze.from_csv
        case _:
            raise argparse.ArgumentTypeError(f"unknown maze type: {typ!r}")
    if arg == '-':
        return text_parser(sys.stdin.read())
    if typ and not os.path.exists(arg):
        return text_parser(arg)
    return file_parser(arg)  # type: ignore


def maze(arg: str) -> Maze:
    """Maze type for argparse."""
    try:
        return __maze(arg)
    except Exception as error:
        raise argparse.ArgumentTypeError(str(error))


def position(arg: str) -> tuple[int, int]:
    """position type for argparse."""
    if m := re.fullmatch(r'(?P<open>\()?\s*(?P<row>\d+)\s*,\s*(?P<col>\d+)\s*(?(open)\)|)', arg):
        return (int(m['row']), int(m['col']))
    raise ValueError


def position_set(arg: str) -> Set[tuple[int, int]]:
    """position type for argparse."""
    return {position(pos) for pos in arg.split(':')}


def direction(arg: str) -> Direction:
    """direction type for argparse."""
    return Direction.from_str(arg)


class Preset(TypedDict, total=True):
    """A maze + starting point + goals preset."""
    file: str
    start_pos: tuple[int, int]
    start_direction: Direction
    goals: Set[tuple[int, int]]


_PRESET_KEYS = frozenset(Preset.__annotations__)


class LoadPreset(argparse.Action):
    """Load a maze + starting point + goals preset.
    """

    def __init__(  # pylint: disable=too-many-arguments
            self,
            option_strings: Sequence[str],
            dest: str,
            default: str | None = None,
            required: bool = False,
            help: str | None = None,  # pylint: disable=redefined-builtin
            metavar: str | tuple[str, ...] | None = None,
            presets_file: str = 'presets.toml',
    ) -> None:
        self.presets: dict[str, Preset] | None = None
        self.presets_file = presets_file
        self.maze_dest, self.start_pos_dest, self.start_direction_dest, self.goal_dest = map(str.strip, dest.split(','))
        super().__init__(
            option_strings=option_strings,
            dest=self.maze_dest,
            default=default,
            required=required,
            help=help,
            metavar=metavar,
        )

    def _load_presets(self, force: bool = False):
        if self.presets and not force:
            return

        def _translate_pos(obj: object, main_name: str, name: str) -> tuple[int, int]:
            if not isinstance(obj, list | tuple):
                raise TypeError(f"in {main_name}: invalid type for '{name}': {type(obj).__name__}")
            if len(obj) != 2:
                raise TypeError(f"in {main_name}: invalid type for '{name}': expected 2 items, found {len(obj)}")
            if not all(isinstance(v, int) for v in obj):
                raise TypeError(
                    f"in {main_name}: invalid type for '{name}': "
                    f"expected ints, found {', '.join(type(v).__name__ for v in obj)}"
                )
            return tuple(obj)

        def _translate(d: dict[str, object], name: str) -> Preset:
            keys = frozenset(d)
            if _PRESET_KEYS != keys:
                if missing := _PRESET_KEYS - keys:
                    raise TypeError(f"in {name}: missing keys: {', '.join(missing)}")
                if extra := keys - _PRESET_KEYS:
                    raise TypeError(f"in {name}: unexpected keys: {', '.join(extra)}")

            if not isinstance(file := d.get('file'), str):
                raise TypeError(f"invalid type for 'file': {type(file).__name__}")

            start_pos = _translate_pos(d.get('start_pos'), name, 'start_pos')

            if not isinstance(start_direction := d.get('start_direction'), str | Direction):
                raise TypeError(f"invalid type for 'start_direction': {type(start_direction).__name__}")
            if isinstance(start_direction, str):
                start_direction = Direction.from_str(start_direction)

            if not isinstance(goals := d.get('goals'), list | Set):
                raise TypeError(f"invalid type for 'goals': {type(goals).__name__}")
            goals = {_translate_pos(g, name, f'goals[{i}]') for i, g in enumerate(goals)}
            return {'file': file, 'start_pos': start_pos, 'start_direction': start_direction, 'goals': goals}

        with open(self.presets_file, encoding="utf-8") as presets_file:
            raw_presets = json.load(presets_file)
        if not isinstance(raw_presets, dict):
            raise TypeError(f"invalid presets main object type: {type(raw_presets).__name__}")
        self.presets = {
            name: _translate(info, name)
            for name, info in raw_presets.items()
        }

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string not in self.option_strings:
            return

        try:
            self._load_presets()
        except Exception as err:
            raise argparse.ArgumentError(self, f"failed to load presets: {err!s}")
        assert self.presets is not None

        if values not in self.presets:
            raise argparse.ArgumentError(self, f"unknown preset: {values}")

        preset = self.presets[values]
        try:
            setattr(namespace, self.maze_dest, maze(preset['file']))
        except Exception as err:
            raise argparse.ArgumentError(self, str(err))
        setattr(namespace, self.start_pos_dest, preset['start_pos'])
        setattr(namespace, self.start_direction_dest, preset['start_direction'])
        setattr(namespace, self.goal_dest, preset['goals'])


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
