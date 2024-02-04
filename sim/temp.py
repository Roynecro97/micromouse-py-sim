from __future__ import annotations

import operator

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pygame

from .maze import Maze, Walls


@dataclass
class MazeTile:
    """Represents a tile in the maze with walls around it."""
    row: int
    col: int
    walls: Walls
    wall_thickness: int = 1
    color: str = 'green'

    def draw(self, screen, tile_size: int, offset: Tuple[int, int] = (0, 0)):
        x = self.col * tile_size + offset[0]
        y = self.row * tile_size + offset[1]
        color = pygame.Color(self.color)
        if Walls.NORTH in self.walls:
            pygame.draw.line(screen, color, (x, y), (x + tile_size, y), self.wall_thickness)
        if Walls.EAST in self.walls:
            pygame.draw.line(screen, color, (x + tile_size, y), (x + tile_size, y + tile_size), self.wall_thickness)
        if Walls.SOUTH in self.walls:
            pygame.draw.line(screen, color, (x, y + tile_size), (x + tile_size, y + tile_size), self.wall_thickness)
        if Walls.WEST in self.walls:
            pygame.draw.line(screen, color, (x, y), (x, y + tile_size), self.wall_thickness)


class MazeRenderer:
    """Class for rendering a maze on screen"""

    def __init__(self, maze: Maze, wall_thickness=1, wall_color='green'):
        self.tiles = [MazeTile(row, col, walls, wall_thickness, wall_color) for row, col, walls in maze]

    def draw(self, screen: pygame.surface.Surface, tile_size: int, offset: Tuple[int, int] = (0, 0)):
        """Draw the maze on screen

        Args:
            screen (_type_): _description_
            tile_size (int): _description_
            offset (Tuple[int, int], optional): _description_. Defaults to (0, 0).
            wall_thickness (int, optional): _description_. Defaults to 1.
            wall_color (str, optional): _description_. Defaults to 'green'.
        """
        for tile in self.tiles:
            tile.draw(screen, tile_size, offset)


class Robot:
    def __init__(self, start_pos: Tuple[int, int], start_heading: float = 0):
        self.pos = start_pos
        self.heading = start_heading * np.pi / 180

    def move_forward(self):
        self.pos = (self.pos[0] + np.cos(self.heading), self.pos[1] + np.sin(self.heading))

    def draw(self, screen: pygame.surface.Surface, offset: Tuple[int, int],
             tile_size: int, body_color: pygame.Color, pointing_color: pygame.Color):
        size = 40
        robot_pos = tuple(map(operator.add, offset, (tile_size // 2 + self.pos[0] * tile_size, tile_size // 2 + self.pos[1] * tile_size)))
        pygame.draw.circle(screen, body_color, robot_pos, size)
        heading_point = (robot_pos[0] + size * np.cos(self.heading), robot_pos[1] + size * np.sin(self.heading))
        pygame.draw.line(screen, pointing_color, robot_pos, heading_point, 5)


def add_text(screen: pygame.surface.Surface, text: str, size: int, center: Tuple[int, int], color='white'):
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
    maze = Maze.from_file('mazes/simple.maze')
    maze_renderer = MazeRenderer(maze)
    robot_maze = Maze.empty(*maze.size)
    robot_maze.add_walls(0, 0, maze[0, 0])
    robot_maze_renderer = MazeRenderer(robot_maze)

    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    pygame.display.set_caption('Maze test')

    full_maze_offset = (20, 100)
    robot_maze_relative_to_full_offset = (700, 0)

    tile_size = 100
    robot = Robot((0, 0), 90)

    while True:
        screen.fill("black")
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            robot.move_forward()
        if keys[pygame.K_a]:
            pass
        if keys[pygame.K_d]:
            pass
        if keys[pygame.K_q]:
            break

        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            break
        if event.type == pygame.KEYDOWN or event.type == pygame.KEYUP:
            pass

        add_text(screen, 'Full Maze', 40, (250, 40))
        add_text(screen, 'Robot View', 40, (970, 40))
        maze_renderer.draw(screen, tile_size, full_maze_offset)
        robot_maze_renderer.draw(screen, tile_size, tuple(map(operator.add, full_maze_offset, robot_maze_relative_to_full_offset)))
        robot.draw(screen, full_maze_offset, tile_size, 'purple', 'gray')
        robot.draw(screen, tuple(map(operator.add, full_maze_offset, robot_maze_relative_to_full_offset)), tile_size, 'purple', 'gray')
        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    _main()
