from __future__ import annotations

from os import environ

from typing import Iterable

import numpy as np

from .maze import Maze, Walls, Direction, RelativeDirection
from .simulator import Simulator, SimulationStatus, random_robot, wall_follower_robot

# Disable the prompt triggered by importing `pygame`.
# autopep8: off
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame  # pylint: disable=wrong-import-order,wrong-import-position
# autopep8: on


class MazeRenderer:
    """Class for rendering a maze on screen"""
    tile_size: int = 100
    wall_thickness: int = 3
    wall_color: str = 'red'
    robot_size: int = 80
    robot_main_color: str = 'blue'
    robot_second_color: str = 'yellow'

    goal_color: str = 'green'

    @classmethod
    def draw(
            cls,
            screen: pygame.surface.Surface,
            maze: Maze,
            offset: tuple[int, int] = (0, 0),
            goal_cells: Iterable[tuple[int, int]] = (),
            robot_pos: tuple[int, int] = (0, 0),
            robot_direction: Direction = Direction.NORTH,
    ):
        """Draw the maze on screen.

        Args:
            screen (pygame.surface.Surface): screen to draw on.
            maze (Maze): maze to draw.
            offset (tuple[int, int], optional): offset for maze in pixels from (0, 0). Defaults to (0, 0).
            robot_pos (tuple[int, int], optional): position of robot in maze (row, col). Defaults to (0, 0).
            robot_direction (Direction): direction robot is facing. Defaults to north.
        """
        color = pygame.Color(cls.wall_color)
        for row, col, walls in maze:
            x = col * cls.tile_size + offset[0]
            y = row * cls.tile_size + offset[1]
            for cell in goal_cells:
                if row == cell[0] and col == cell[1]:
                    pygame.draw.rect(screen, cls.goal_color, pygame.Rect(x, y, cls.tile_size, cls.tile_size))

            if Walls.NORTH in walls:
                pygame.draw.line(screen, color, (x, y), (x + cls.tile_size, y), cls.wall_thickness)
            if Walls.EAST in walls:
                pygame.draw.line(screen, color, (x + cls.tile_size, y), (x + cls.tile_size, y + cls.tile_size), cls.wall_thickness)
            if Walls.SOUTH in walls:
                pygame.draw.line(screen, color, (x, y + cls.tile_size), (x + cls.tile_size, y + cls.tile_size), cls.wall_thickness)
            if Walls.WEST in walls:
                pygame.draw.line(screen, color, (x, y), (x, y + cls.tile_size), cls.wall_thickness)

            if row == robot_pos[1] and col == robot_pos[0]:
                cls.draw_robot(screen, (x, y), robot_direction)

    @classmethod
    def draw_robot(cls, screen: pygame.surface.Surface, robot_pos: tuple[int, int], robot_direction: Direction):
        """Draw robot on screen.

        Args:
            screen (pygame.surface.Surface): screen to draw on.
            robot_pos (tuple[int, int]): position of robot in maze (in pixels).
            robot_direction (Direction): direction robot is facing
            offset (tuple[int, int], optional): offset for maze in pixels from (0, 0). Defaults to (0, 0).
        """
        robot_radius = cls.robot_size // 2
        robot_pos = (robot_pos[0] + cls.tile_size // 2, robot_pos[1] + cls.tile_size // 2)
        pygame.draw.circle(screen, cls.robot_main_color, robot_pos, robot_radius)
        heading_point = (robot_pos[0] + robot_radius * np.cos(robot_direction.to_radians()),
                         robot_pos[1] + robot_radius * np.sin(robot_direction.to_radians()))
        pygame.draw.line(screen, cls.robot_second_color, robot_pos, heading_point, 5)


class Robot:
    def __init__(self, start_pos: tuple[int, int], start_heading: Direction = Direction.NORTH):
        self.pos: tuple[int, int] = start_pos
        self.heading: Direction = start_heading

    def move_forward(self):
        self.pos = (self.pos[0] + round(np.cos(self.heading.to_radians())),
                    self.pos[1] + round(np.sin(self.heading.to_radians())))

    def move_reverse(self):
        self.pos = (self.pos[0] - round(np.cos(self.heading.to_radians())),
                    self.pos[1] - round(np.sin(self.heading.to_radians())))

    def turn_left(self):
        self.heading = self.heading.turn_left()

    def turn_right(self):
        self.heading = self.heading.turn_right()

    def try_move_forward(self, maze: Maze):
        self.pos = (self.pos[0] + round(np.cos(self.heading.to_radians())),
                    self.pos[1] + round(np.sin(self.heading.to_radians())))


def draw_text(screen: pygame.surface.Surface, text: str, size: int, center: tuple[int, int], color='white'):
    font = pygame.font.Font(pygame.font.get_default_font(), size)
    text = font.render(text, True, color)
    textRect = text.get_rect()
    textRect.center = center
    screen.blit(text, textRect)


def _main():
    #     _ _ _ _ _
    #    | |  _   _|
    #    | |_ _|   |
    #    |       | |
    #    | |_ _|_  |
    #    |_|_ _ _ _|
    # maze = Maze.from_maz(
    #     b'\x0B\x09\x05\x01\x07'
    #     b'\x0A\x0C\x07\x08\x03'
    #     b'\x08\x01\x01\x02\x0A'
    #     b'\x0A\x0C\x06\x0C\x02'
    #     b'\x0E\x0D\x05\x05\x06'
    # )
    sim = Simulator(
        alg=wall_follower_robot(RelativeDirection.RIGHT),
        maze=Maze.from_file('mazes/simple.maze'),
        begin=(0, 0, Direction.SOUTH),
        end={(1, 2)},
    )

    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    pygame.display.set_caption('Maze test')

    full_maze_offset = (20, 100)
    robot_maze_offset = (720, 100)

    robot = Robot((0, 0), Direction.NORTH)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    sim.step()
                    print(f"pygame: after step - {sim.maze[sim.robot_pos[:-1]]=} {sim.robot_maze[sim.robot_pos[:-1]]=}")
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

        screen.fill("black")
        draw_text(screen, 'Full Maze', 40, (250, 40))
        draw_text(screen, 'Robot View', 40, (970, 40))
        robot_y, robot_x, robot_heading = sim.robot_pos
        MazeRenderer.draw(screen, sim.maze, full_maze_offset, sim.end, (robot_x, robot_y), robot_heading)
        MazeRenderer.draw(screen, sim.robot_maze, robot_maze_offset, sim.end, (robot_x, robot_y), robot_heading)
        pygame.display.update()

    pygame.quit()


if __name__ == '__main__':
    _main()
