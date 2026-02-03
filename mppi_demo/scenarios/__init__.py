"""
Scenarios Package

Predefined test scenarios for MPPI path planning.
Each scenario module provides a function to create the environment.
"""

from .u_trap import create_u_trap_scenario
from .narrow_passage import create_narrow_passage_scenario
from .christmas_market import create_christmas_market_environment

__all__ = [
    'create_u_trap_scenario',
    'create_narrow_passage_scenario',
    'create_christmas_market_environment',
]
