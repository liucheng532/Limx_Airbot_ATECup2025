# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Task registration module to avoid circular imports.
This module should be imported after the task_registry is fully initialized.
"""

from legged_gym.utils.task_registry import task_registry
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.anymal_c.anymal import Anymal
from legged_gym.envs.anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
from legged_gym.envs.anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
from legged_gym.envs.anymal_b.anymal_b_config import AnymalBRoughCfg, AnymalBRoughCfgPPO
from legged_gym.envs.cassie.cassie import Cassie
from legged_gym.envs.cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO

def register_legged_gym_tasks():
    """Register all legged_gym tasks to avoid circular imports."""
    task_registry.register("anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO())
    task_registry.register("anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO())
    task_registry.register("anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO())
    task_registry.register("a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO())
    task_registry.register("cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO())

# Auto-register tasks when this module is imported
register_legged_gym_tasks()



