"""Various mouse robot algorithms.

Simple:
+ Idle
+ Random
+ Wall Follower (left/right)
+ Predetermined [TODO]

Advanced:
+ Flood Fill:
  + Simple - no diagonals, shortest path
  + Weighted - no diagonals, fastest path by time [TODO]
+ BFS [TODO]
+ DFS [TODO]
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from . import const, flood_fill, idle, random, wall_follower
from .const import predetermined_robot
from .flood_fill import simple_flood_fill, basic_weighted_flood_fill, thourough_flood_fill
from .idle import idle_robot
from .random import random_robot
from .utils import Action, RobotState
from .wall_follower import wall_follower_robot

if TYPE_CHECKING:
    from .utils import Algorithm, Robot
