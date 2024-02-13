"""Temp file currently running the GUI
"""

from __future__ import annotations

from datetime import datetime, timedelta
from os import environ
from typing import Iterable

import operator
import numpy as np

from .maze import Maze, Walls, Direction, RelativeDirection
from .simulator import Simulator, SimulationStatus, random_robot, wall_follower_robot, idle_robot, simple_flood_fill

# Disable the prompt triggered by importing `pygame`.
# autopep8: off
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame  # pylint: disable=wrong-import-order,wrong-import-position
# autopep8: on


class GUIRenderer:
    """Class for rendering a maze on screen"""
    wall_thickness: int = 5
    wall_color: str = 'red'
    robot_main_color: str = 'blue'
    robot_second_color: str = 'yellow'
    goal_color: str = 'green'

    @classmethod
    def draw_maze(
            cls,
            screen: pygame.surface.Surface,
            maze: Maze,
            tile_size: int,
            offset: tuple[int, int] = (0, 0),
            goal_cells: Iterable[tuple[int, int]] = (),
            robot_pos: tuple[int, int] = (0, 0),
            robot_direction: Direction = Direction.NORTH,
    ):
        """Draw the maze on screen.

        Args:
            screen (pygame.surface.Surface): screen to draw on.
            maze (Maze): maze to draw.
            tile_size (int): the size of a maze's single tile
            offset (tuple[int, int], optional): offset for maze in pixels from (0, 0). Defaults to (0, 0).
            robot_pos (tuple[int, int], optional): position of robot in maze (row, col). Defaults to (0, 0).
            robot_direction (Direction): direction robot is facing. Defaults to north.
        """
        color = pygame.Color(cls.wall_color)
        for row, col, walls in maze:
            x = col * tile_size + offset[0]
            y = row * tile_size + offset[1]
            for cell in goal_cells:
                if (row, col) == cell:
                    pygame.draw.rect(screen, cls.goal_color, pygame.Rect(x, y, tile_size, tile_size))

            if Walls.NORTH in walls:
                pygame.draw.line(screen, color, (x, y), (x + tile_size, y), cls.wall_thickness)
            if Walls.EAST in walls:
                pygame.draw.line(screen, color, (x + tile_size, y), (x + tile_size, y + tile_size), cls.wall_thickness)
            if Walls.SOUTH in walls:
                pygame.draw.line(screen, color, (x, y + tile_size), (x + tile_size, y + tile_size), cls.wall_thickness)
            if Walls.WEST in walls:
                pygame.draw.line(screen, color, (x, y), (x, y + tile_size), cls.wall_thickness)

            if (col, row) == robot_pos:
                cls.draw_robot(screen, (x, y), robot_direction, tile_size)

    @classmethod
    def draw_robot(cls, screen: pygame.surface.Surface, robot_pos: tuple[int, int], robot_direction: Direction, tile_size: int):
        """Draw robot on screen.

        Args:
            screen (pygame.surface.Surface): screen to draw on.
            robot_pos (tuple[int, int]): position of robot in maze (in pixels).
            robot_direction (Direction): direction robot is facing
            offset (tuple[int, int], optional): offset for maze in pixels from (0, 0). Defaults to (0, 0).
            tile_size (int): the size of a maze's single tile
        """
        robot_radius = (tile_size * 0.8) // 2
        robot_pos = (robot_pos[0] + tile_size // 2, robot_pos[1] + tile_size // 2)
        pygame.draw.circle(screen, cls.robot_main_color, robot_pos, robot_radius)
        heading_point = (robot_pos[0] + robot_radius * np.cos(robot_direction.to_radians()),
                         robot_pos[1] + robot_radius * np.sin(robot_direction.to_radians()))
        pygame.draw.line(screen, cls.robot_second_color, robot_pos, heading_point, 1)

    @classmethod
    def draw_text(cls, screen: pygame.surface.Surface, text: str, size: int, center: tuple[int, int], color='white'):
        """Draw text on screen.

        Args:
            screen (pygame.surface.Surface): screen to draw on.
            text (str): text to draw.
            size (int): size of text.
            center (tuple[int, int]): center of text on screen (x, y).
            color (str, optional): text's color. Defaults to 'white'.
        """
        font = pygame.font.Font(pygame.font.get_default_font(), size)
        text = font.render(text, True, color)
        textRect = text.get_rect()
        textRect.center = center
        screen.blit(text, textRect)

    @classmethod
    def _scale(cls, maze_width: int, maze_height: int):
        """Calculate sizes and offsets according to screen size

        Args:
            maze_width (int): _description_
            maze_height (int): _description_
        """
        screen_width, screen_height = pygame.display.get_surface().get_size()
        tile_size = min(screen_width // (2 * maze_width + 3), screen_height // (maze_height + 2))
        half_tile = tile_size // 2
        full_maze_offset = (tile_size, tile_size)
        full_maze_center_offset = add_tuples_elementwise(full_maze_offset, ((maze_width * half_tile, 0)))
        robot_maze_offset = add_tuples_elementwise(full_maze_offset, ((maze_width + 1) * tile_size, 0))
        robot_maze_center_offset = add_tuples_elementwise(robot_maze_offset, ((maze_width * half_tile, 0)))
        return tile_size, half_tile, full_maze_offset, full_maze_center_offset, robot_maze_offset, robot_maze_center_offset

    @classmethod
    def main_loop(cls, screen: pygame.surface.Surface, sim: Simulator):
        """Main GUI loop. This function will stay in a loop.

        Args:
            screen (pygame.surface.Surface): screen to draw on.
            sim (Simulator): the micromouse simulator to draw.
        """
        step_min_delay = timedelta(milliseconds=100)
        step_delay = timedelta(seconds=0.5)
        last_step = datetime.now()

        tile_size, half_tile, full_maze_offset, full_maze_center_offset, robot_maze_offset, robot_maze_center_offset = cls._scale(
            sim.maze.width, sim.maze.height)

        while True:
            step = False
            now = datetime.now()

            if pygame.key.get_pressed()[pygame.K_SPACE]:
                step = now - last_step >= step_min_delay

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.VIDEORESIZE:
                    tile_size, half_tile, full_maze_offset, full_maze_center_offset, robot_maze_offset, robot_maze_center_offset = cls._scale(
                        sim.maze.width, sim.maze.height)
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_r, pygame.K_l, pygame.K_f):
                        match (event.key, event.mod & pygame.KMOD_SHIFT != 0):
                            case (pygame.K_r, True): sim.restart(random_robot)
                            case (pygame.K_r, _): sim.restart(wall_follower_robot(RelativeDirection.RIGHT))
                            case (pygame.K_l, _): sim.restart(wall_follower_robot(RelativeDirection.LEFT))
                            case (pygame.K_f, _): sim.restart(simple_flood_fill)
                        step = False
                        last_step = now
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        return

            if now - last_step >= step_delay:
                step = True

            if sim.status in (SimulationStatus.ERROR, SimulationStatus.FINISHED):
                step = False

            if step:
                sim.step()
                last_step = now
                print(f"GUI: after step - {sim.maze[sim.robot_pos[:-1]]=} {sim.robot_maze[sim.robot_pos[:-1]]=}")

            screen.fill("black")
            cls.draw_text(screen, 'Full Maze', half_tile, add_tuples_elementwise(full_maze_center_offset, (0, -half_tile)))
            cls.draw_text(screen, 'Robot View', half_tile, add_tuples_elementwise(robot_maze_center_offset, (0, -half_tile)))
            robot_y, robot_x, robot_heading = sim.robot_pos
            cls.draw_maze(screen, sim.maze, tile_size, full_maze_offset, sim.end, (robot_x, robot_y), robot_heading)
            cls.draw_maze(screen, sim.robot_maze, tile_size, robot_maze_offset, sim.end, (robot_x, robot_y), robot_heading)
            pygame.display.update()


def add_tuples_elementwise[T, *Ts](t1: tuple[T, *Ts], t2: tuple[T, *Ts]) -> tuple[T, *Ts]:
    """Add two tuples elementwise.
    """
    return tuple(map(operator.add, t1, t2))


def _main():
    sim = Simulator(
        alg=idle_robot,
        maze=Maze.from_file('mazes/simple.maze'),
        begin=(0, 0, Direction.SOUTH),
        end={(1, 2)},
    )

    pygame.init()
    screen_size = (1280, 720)
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption('Maze test')

    GUIRenderer.main_loop(screen, sim)
    pygame.quit()

if __name__ == '__main__':
    _main()
