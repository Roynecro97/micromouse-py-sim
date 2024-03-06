"""GUI elements and running the GUI
"""
# pylint gives too many false-positive "no member" for pygame attributes
# pylint: disable=no-member

from __future__ import annotations

import math

from datetime import datetime, timedelta
from os import environ
from typing import Iterable, NamedTuple, Self

from .maze import Direction, ExtraCellInfo, ExtendedMaze, Maze, RelativeDirection, Walls
from .robots import idle_robot, random_robot, simple_flood_fill, wall_follower_robot
from .simulator import SimulationStatus, Simulator

# Disable the prompt triggered by importing `pygame`.
# autopep8: off
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame as pg  # pylint: disable=wrong-import-order,wrong-import-position
import pygame_gui    # pylint: disable=wrong-import-order,wrong-import-position
# autopep8: on


ROBOTS = {
    'Idle': idle_robot,
    'Random': random_robot,
    'Left Wall Follower': wall_follower_robot(RelativeDirection.LEFT),
    'Right Wall Follower': wall_follower_robot(RelativeDirection.RIGHT),
    'Flood Fill': simple_flood_fill,
}


class Position(NamedTuple):
    """Represents a position on screen."""
    row: int
    col: int

    def __add__(self, other: object) -> Self:
        if not isinstance(other, type(self)):
            return NotImplemented
        return type(self)(self.row + other.row, self.col + other.col)


class GUIRenderer:  # pylint: disable=too-many-instance-attributes
    """Class for rendering the GUI on screen"""

    def __init__(self, sim: Simulator):
        """Initialize GUI

        Args:
            sim (Simulator): The simulation to show.
        """
        pg.init()
        initial_screen_size = (1280, 720)
        screen = pg.display.set_mode(initial_screen_size, pg.RESIZABLE)
        pg.display.set_caption('Micromouse')
        self.screen = screen
        self.wall_thickness = 5
        self.wall_color = 'red'
        self.robot_main_color = 'blue'
        self.robot_second_color = 'yellow'
        self.goal_color = 'green'
        self.heatmap_colors = ['blue', 'cyan', 'green', 'yellow', 'orange', 'red', 'brown']
        self.ui_manager = pygame_gui.UIManager((screen.get_width(), screen.get_height()))
        self.start_button = pygame_gui.elements.UIButton(relative_rect=pg.Rect((0, 0), (0, 0)), text='Start', manager=self.ui_manager)
        self.start_button.tool_tip_text = "Shortcut: 's'"
        self.step_button = pygame_gui.elements.UIButton(relative_rect=pg.Rect((0, 0), (0, 0)), text='Step', manager=self.ui_manager)
        self.step_button.tool_tip_text = "Shortcut: 'n'"
        self.robot_dropdown = pygame_gui.elements.UIDropDownMenu(list(ROBOTS), 'Idle', pg.Rect((0, 0), (0, 0)), self.ui_manager)

        self.sim_auto_step = True
        self.sim = sim

        self.screen_width, self.screen_height = initial_screen_size
        self.text_size = 4 * min(self.screen_width, self.screen_height) // 100
        self.tile_size = min(self.screen_width // (2 * self.sim.maze.width + 3), self.screen_height // (self.sim.maze.height + 4))
        self.half_tile = self.tile_size // 2
        self.full_maze_offset = Position(self.tile_size, 3 * self.tile_size)
        self.full_maze_center = self.full_maze_offset + Position(self.sim.maze.width * self.half_tile, 0)
        self.robot_maze_offset = self.full_maze_offset + Position((self.sim.maze.width + 1) * self.tile_size, 0)
        self.robot_maze_center = self.robot_maze_offset + Position(self.sim.maze.width * self.half_tile, 0)

    def run(self):
        """Run the GUI
        """
        # self.configure_simulator()
        self.scale()
        self.main_loop()
        pg.quit()

    def get_selected_robot(self):
        """Fetch the selected robot.

        Returns:
            Robot: The chosen robot
        """
        return ROBOTS[self.robot_dropdown.selected_option]

    def get_selected_maze(self):
        """Fetch the selected preset for maze.

        Returns:
            Maze: The selected maze.
        """
        return ('mazes/semifinal.maze', (15, 0, Direction.EAST), {(8, 8), (7, 8), (8, 7), (7, 7)})

    def configure_simulator(self):
        """Configure the simulator with the chosen maze preset and robot."""
        self.sim_auto_step = True
        maze_file, begin, end = self.get_selected_maze()
        self.sim = Simulator(
            alg=self.get_selected_robot(),
            maze=Maze.from_file(maze_file),
            begin=begin,
            end=end,
        )
        self.scale()

    def process_event(self, event: pg.Event):
        """Process pygame event.

        Args:
            event (pg.Event): the event to process.
        """
        self.ui_manager.process_events(event)
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.start_button:
                self.sim_auto_step = not self.sim_auto_step
            elif event.ui_element == self.step_button:
                self.sim_auto_step = False
                self.sim.step()
        elif event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
            if event.ui_element == self.robot_dropdown:
                self.sim_auto_step = False
                self.sim.restart(self.get_selected_robot())
        elif event.type == pg.KEYDOWN:
            if event.key in (pg.K_f, pg.K_l, pg.K_n, pg.K_r, pg.K_s):
                match (event.key, event.mod & pg.KMOD_SHIFT != 0):
                    case (pg.K_f, _): self.sim.restart(simple_flood_fill)
                    case (pg.K_l, _): self.sim.restart(wall_follower_robot(RelativeDirection.LEFT))
                    case (pg.K_n, _):
                        self.sim_auto_step = False
                        self.sim.step()
                    case (pg.K_r, True): self.sim.restart(random_robot)
                    case (pg.K_r, _): self.sim.restart(wall_follower_robot(RelativeDirection.RIGHT))
                    case (pg.K_s, _): self.sim_auto_step = not self.sim_auto_step

    def update(self, time_delta: float):
        """Update GUI elements and draw them on screen.

        Args:
            time_delta (float): The time passed since the last call to update, in seconds.
        """
        self.start_button.set_text('Stop' if self.sim_auto_step else 'Start')

        self.ui_manager.update(time_delta)
        self.ui_manager.draw_ui(self.screen)
        pg.display.update()

    def draw_maze(
            self,
            maze: ExtendedMaze,
            tile_size: int,
            offset: tuple[int, int] = (0, 0),
            goal_cells: Iterable[tuple[int, int]] = (),
            robot_pos: tuple[int, int] = (0, 0),
            robot_direction: Direction = Direction.NORTH,
            heatmap: bool = False,
    ):  # pylint: disable=too-many-arguments,too-many-locals
        """Draw the maze on screen.

        Args:
            maze (Maze): maze to draw.
            tile_size (int): the size of a maze's single tile
            offset (tuple[int, int], optional): offset for maze in pixels from (0, 0). Defaults to (0, 0).
            robot_pos (tuple[int, int], optional): position of robot in maze (row, col). Defaults to (0, 0).
            robot_direction (Direction): direction robot is facing. Defaults to north.
        """
        color = pg.Color(self.wall_color)
        for row, col, walls in maze:
            x = col * tile_size + offset[0]
            y = row * tile_size + offset[1]
            current_pos = Position(x, y)
            extra_info = maze.extra_info[row, col]

            if heatmap:
                self.draw_heatmap(extra_info, current_pos, tile_size)

            for cell in goal_cells:
                if (row, col) == cell:
                    pg.draw.rect(self.screen, self.goal_color, pg.Rect(x, y, tile_size, tile_size))

            self.draw_tile_color(extra_info, current_pos, tile_size)

            if (col, row) == robot_pos:
                self.draw_robot(current_pos, robot_direction, tile_size)

            if extra_info.weight is not None:
                self.draw_text(str(extra_info.weight), tile_size // 2, current_pos + Position(tile_size // 2, tile_size // 2))

            line_end = self.wall_thickness // 2
            if Walls.NORTH in walls:
                pg.draw.line(self.screen, color, (x - line_end, y), (x + line_end + tile_size, y), self.wall_thickness)
            if Walls.EAST in walls:
                pg.draw.line(self.screen, color, (x + tile_size, y - line_end),
                             (x + tile_size, y + line_end + tile_size), self.wall_thickness)
            if Walls.SOUTH in walls:
                pg.draw.line(self.screen, color, (x - line_end, y + tile_size),
                             (x + line_end + tile_size, y + tile_size), self.wall_thickness)
            if Walls.WEST in walls:
                pg.draw.line(self.screen, color, (x, y - line_end), (x, y + line_end + tile_size), self.wall_thickness)

    def draw_heatmap(self, info: ExtraCellInfo, cell_pos: Position, tile_size: int):
        """Fill cell color by visit count to produce a heatmap.

        Args:
            info (ExtraCellInfo): cell info to draw.
            cell_pos (Position): cell position in pixels.
            tile_size (int): the size of a maze's single tile.
        """
        color = None
        if info.color is not None:
            color = pg.Color(info.color)
        elif info.visited > 0:
            color = pg.Color(self.heatmap_colors[min(info.visited, len(self.heatmap_colors)) - 1])

        if color is not None:
            alpha_surface = pg.Surface((tile_size, tile_size), pg.SRCALPHA)
            alpha_surface.fill((color.r, color.g, color.b, 200))
            pg.draw.rect(alpha_surface, color, pg.Rect(cell_pos.row, cell_pos.col, tile_size, tile_size))
            self.screen.blit(alpha_surface, cell_pos)

    def draw_tile_color(self, info: ExtraCellInfo, cell_pos: Position, tile_size: int):
        """Fill cell with info color.

        Args:
            info (ExtraCellInfo): cell info to draw.
            cell_pos (Position): cell position in pixels.
            tile_size (int): the size of a maze's single tile.
        """
        if info.color is not None:
            alpha_surface = pg.Surface((tile_size, tile_size), pg.SRCALPHA)
            color = pg.Color(info.color)
            alpha_surface.fill((color.r, color.g, color.b, 128))
            pg.draw.rect(alpha_surface, color, pg.Rect(cell_pos.row, cell_pos.col, tile_size, tile_size))
            self.screen.blit(alpha_surface, cell_pos)

    def draw_robot(self, robot_pos: Position, robot_direction: Direction, tile_size: int):
        """Draw robot on screen.

        Args:
            robot_pos (Position): position of robot in maze (in pixels).
            robot_direction (Direction): direction robot is facing.
            tile_size (int): the size of a maze's single tile.
        """
        robot_radius = (tile_size * 0.8) // 2
        robot_pos = robot_pos + Position(tile_size // 2, tile_size // 2)
        pg.draw.circle(self.screen, self.robot_main_color, robot_pos, robot_radius)
        heading_point = robot_pos + Position(round(robot_radius * math.cos(robot_direction.to_radians())),
                                             round(robot_radius * math.sin(robot_direction.to_radians())))
        pg.draw.line(self.screen, self.robot_second_color, robot_pos, heading_point, 1)

    def draw_text(self, text: str, size: int, center: Position, color='white'):
        """Draw text on screen.

        Args:
            text (str): text to draw.
            size (int): size of text.
            center (Position): center of text on screen (x, y).
            color (str, optional): text's color. Defaults to 'white'.
        """
        font = pg.font.Font(pg.font.get_default_font(), size)
        render_text = font.render(text, True, color)
        text_rect = render_text.get_rect()
        text_rect.center = center
        self.screen.blit(render_text, text_rect)

    def scale(self):
        """Calculate sizes and offsets of GUI elements according to screen size."""
        self.screen_width, self.screen_height = pg.display.get_surface().get_size()
        self.ui_manager.set_window_resolution((self.screen_width, self.screen_height))
        self.text_size = 4 * min(self.screen_width, self.screen_height) // 100
        self.tile_size = min(self.screen_width // (2 * self.sim.maze.width + 3), self.screen_height // (self.sim.maze.height + 5))
        self.half_tile = self.tile_size // 2
        self.full_maze_offset = Position(self.tile_size, 4 * self.tile_size)
        self.full_maze_center = self.full_maze_offset + Position(self.sim.maze.width * self.half_tile, 0)
        self.robot_maze_offset = self.full_maze_offset + Position((self.sim.maze.width + 1) * self.tile_size, 0)
        self.robot_maze_center = self.robot_maze_offset + Position(self.sim.maze.width * self.half_tile, 0)
        self.start_button.set_position((self.screen_width // 2 - 2 * self.tile_size, 2 * self.tile_size))
        self.start_button.set_dimensions((3 * self.tile_size, self.tile_size))
        self.step_button.set_position((self.screen_width // 2 + 2 * self.tile_size, 2 * self.tile_size))
        self.step_button.set_dimensions((3 * self.tile_size, self.tile_size))
        self.robot_dropdown.set_position((self.screen_width // 2, 0))
        self.robot_dropdown.set_dimensions((5 * self.tile_size, self.tile_size))

    def main_loop(self):  # pylint: disable=too-many-locals
        """Main GUI loop. This function will stay in a loop.

        Args:
            screen (pg.surface.Surface): screen to draw on.
            sim (Simulator): the micromouse simulator to draw.
        """
        step_min_delay = timedelta(milliseconds=100)
        step_delay = timedelta(seconds=0.5)
        last_step = datetime.now()
        clock = pg.time.Clock()
        while True:
            time_delta = clock.tick(60) / 1000.0
            step = False
            now = datetime.now()

            if pg.key.get_pressed()[pg.K_SPACE]:
                step = now - last_step >= step_min_delay

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return
                if event.type == pg.VIDEORESIZE:
                    self.scale()
                elif event.type == pg.KEYDOWN:
                    if event.key in (pg.K_q, pg.K_ESCAPE):
                        return

                self.process_event(event)

            if now - last_step >= step_delay:
                step = True

            if self.sim.status in (SimulationStatus.ERROR, SimulationStatus.FINISHED):
                step = False
                self.sim_auto_step = False

            if step and self.sim_auto_step:
                self.sim.step()
                last_step = now
                print(f"GUI: after step - {self.sim.maze[self.sim.robot_pos[:-1]]=} {self.sim.robot_maze[self.sim.robot_pos[:-1]]=}")

            self.screen.fill("black")
            self.draw_text('Full Maze', self.text_size, self.full_maze_center + Position(0, -self.half_tile))
            self.draw_text('Robot View', self.text_size, self.robot_maze_center + Position(0, -self.half_tile))
            robot_y, robot_x, robot_heading = self.sim.robot_pos
            self.draw_maze(self.sim.maze, self.tile_size, self.full_maze_offset,
                           self.sim.end, (robot_x, robot_y), robot_heading, heatmap=True)
            self.draw_maze(self.sim.robot_maze, self.tile_size, self.robot_maze_offset, self.sim.end, (robot_x, robot_y), robot_heading)

            # texts for menus:
            self.draw_text('Algorithm:', self.text_size, Position(self.screen_width // 2 - 110, self.half_tile))

            self.update(time_delta)


def _main():
    sim = Simulator(
        alg=idle_robot,
        maze=Maze.from_file('mazes/simple.maze'),
        begin=(0, 0, Direction.SOUTH),
        end={(1, 2)},
    )

    GUIRenderer(sim).run()


if __name__ == '__main__':
    _main()
