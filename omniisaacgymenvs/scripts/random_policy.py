# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
import torch
import hydra
from omegaconf import DictConfig

from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict

from omniisaacgymenvs.utils.task_util import initialize_task
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames

#    For logging
import pandas as pd
import time
#____________________

@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    headless = cfg.headless
    render = not headless

    env = VecEnvRLGames(headless=headless, sim_device=cfg.device_id)
    # sets seed. if seed is -1 will pick a random one
    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict['seed'] = cfg.seed
    task = initialize_task(cfg_dict, env)

    while env._simulation_app.is_running():
        if env._world.is_playing():
            if env._world.current_time_step_index == 0:
                env._world.reset(soft=True)
            actions = torch.tensor(np.array([env.action_space.sample() for _ in range(env.num_envs)]), device=task.rl_device)

            actions[:, 0] = 0.0   #Thrust
            actions[:, 1] = 0.0   #Roll rate 
            actions[:, 2] = 0.0   #Pitch rate 
            actions[:, 3] = 0.8   #Yaw rate 
            # print("actions = ",actions)
            env._task.pre_physics_step(actions)
            
            env._world.step(render=render)
            env.sim_frame_count += 1
            env._task.post_physics_step()
        else:
            env._world.step(render=render)
    
    print("Simulation finished")
    print("Saving logs...")
    filename = "iris_observation_logs_" + time.strftime("%Y%m%d-%H%M%S") + ".csv"
    df_obs = pd.DataFrame(task.obs_log.cpu().numpy(), columns=["err_x","err_y","err_z","rot_x_1","rot_x_2","rot_x_3","rot_y_1","rot_y_2","rot_y_3","rot_z_1","rot_z_2","rot_z_3","root_linvels_x","root_linvels_y","root_linvels_z","root_angvels_x","root_angvels_y","root_angvels_z","relative_pos0_x","relative_pos0_y","relative_pos0_z","relative_pos1_x","relative_pos1_y","relative_pos1_z","relative_pos2_x","relative_pos2_y","relative_pos2_z","relative_pos3_x","relative_pos3_y","relative_pos3_z"])
    df_obs.to_csv(filename,index=False)
    
    # df_action = pd.DataFrame(task.action_log.cpu().numpy(), columns=["thrust","roll","pitch","yaw"])
    # df_action.to_csv("iris_action_256env_horizontal.csv",index=False)
    
    env._simulation_app.close()




if __name__ == '__main__':
    parse_hydra_configs()

