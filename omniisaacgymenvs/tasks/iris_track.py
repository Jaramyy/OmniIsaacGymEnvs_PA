# from omni.isaac.kit import SimulationApp
# simulation_app = SimulationApp({"headless": False}) # we can also run as headless.

import numpy as np
import torch
from typing import Optional
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.objects import DynamicSphere
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.iris import iris
from omniisaacgymenvs.robots.articulations.views.iris_view import irisView
from omni.isaac.debug_draw import _debug_draw
from omni.isaac.core.utils.viewports import set_camera_view

class irisTask(RLTask):
    def __init__(
            self,
            name,
            sim_config,
            env,
            offset=None
    ) -> None:
        
        from omni.isaac.core.utils.nucleus import get_server_path
        self.server_path = get_server_path()
        if self.server_path is None:
            print("Could not find Isaac Sim server path")
            return

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]
        self.dt = self._task_cfg["sim"]["dt"]
        
        self.mass = 1.2
        self.thrust_to_weight = 12.0   
        self.grav_z = -1.0 * self._task_cfg["sim"]["gravity"][2]

        self._num_observations = 30 # 18 for non future trajectory, 18 + 3*future_traj_steps for future trajectory  
        self._num_actions = 4

        RLTask.__init__(self, name=name, env=env)
        


        self.prop_max_rot = 5.0
        self.number_of_rotors = 4
        self.dimension = 3 #xyz
        self.thrusts = torch.zeros((self._num_envs, self.number_of_rotors, self.dimension), dtype=torch.float32, device=self._device)
        self.torques = torch.zeros((self._num_envs, self.dimension), dtype=torch.float32, device=self._device)

        self.target_positions = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self.target_positions[:, 2] = 1
        self._ball_position = torch.tensor([0, 0, 1.0])
        
        self.actions = torch.zeros((self._num_envs, 4), device=self._device, dtype=torch.float32)
        self.prev_actions = torch.zeros((self._num_envs, 4), device=self._device, dtype=torch.float32) #for 1 previous time step
        self.actions_cmds = torch.zeros((self._num_envs, 4), device=self._device, dtype=torch.float32)

        self.all_indices = torch.arange(self._num_envs, dtype=torch.int32, device=self._device)
    
        self._num_rotors = 4
        self._rotor_constant = torch.tensor([8.54858e-6, 8.54858e-6, 8.54858e-6, 8.54858e-6], device=self._device, dtype=torch.float32) 
        self._rolling_moment_coefficient =  torch.tensor([1e-6, 1e-6, 1e-6, 1e-6], device=self._device, dtype=torch.float32)
        self._rot_dir = torch.tensor([-1, -1, 1, 1], device=self._device, dtype=torch.float32)
        
        self.max_rotor_velocity = torch.tensor([1100, 1100, 1100, 1100], device=self._device, dtype=torch.float32)
        self.relative_poses_x = torch.tensor([0.138,-0.125,0.138,-0.125], device=self._device, dtype=torch.float32)
        self.relative_poses_y = torch.tensor([-0.22,0.22,0.22,-0.22], device=self._device, dtype=torch.float32)

        # Extra info
        self.extras = {}
        
        torch_zeros = lambda: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {"rew_pos": torch_zeros(), 
                             "rew_orient": torch_zeros(), 
                             "rew_effort": torch_zeros(),
                             "rew_spin": torch_zeros(),
                             "raw_dist": torch_zeros(), 
                             "raw_orient": torch_zeros(), 
                             "raw_effort": torch_zeros(),
                             "raw_spin": torch_zeros()}
        
        self.obs_log = torch.empty((1,self._num_observations), dtype=torch.float32, device=self._device)
        # self.error_log = [] #not used for now

        self.future_traj_steps = 4
        self.traj_t0 = 0.0
        self.traj_w = torch.ones(self._num_envs, device=self._device)
        self.traj_c = torch.zeros(self._num_envs, device=self._device)
        
        self.traj_scale = torch.zeros(self._num_envs, 3, device=self._device)
        self.traj_rot = torch.zeros(self._num_envs, 4, device=self._device)
        self.target_pos = torch.zeros(self._num_envs, self.future_traj_steps, 3, device=self.device)
        self.origin = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        self.ref_err = torch.zeros(self.num_envs, 2, device=self.device)

        self.draw = _debug_draw.acquire_debug_draw_interface()
        self.draw.clear_lines()
        self.render = True

        
        return  


    def set_up_scene(self, scene) -> None:
        # self.get_world_scene()
        self.get_iris()
        self.get_target()
        RLTask.set_up_scene(self, scene)
        
        self.central_env_idx = self._env_pos.norm(dim=-1).argmin()
        # central_env_pos = self._env_pos[self.central_env_idx].cpu().numpy()
        # set_camera_view(
        #     eye=central_env_pos + np.asarray(self.cfg.viewer.eye), 
        #     target=central_env_pos + np.asarray(self.cfg.viewer.lookat)
        # )

        self._copters = irisView(prim_paths_expr="/World/envs/.*/iris", name="irisView")
        self._balls = RigidPrimView(prim_paths_expr="/World/envs/.*/ball")
        scene.add(self._copters)
        scene.add(self._balls)

        for i in range(4):
            scene.add(self._copters.physics_rotors[i])
        scene.add(self._copters.physics_bodys)
        return
    
    def get_iris(self):
        self._iris_position = torch.tensor([0.0, 0, 1.05])
        copter = iris(prim_path=self.default_zero_env_path + "/iris", name="iris",translation=self._iris_position)
        self._sim_config.apply_articulation_settings("iris", get_prim_at_path(copter.prim_path),
                                                     self._sim_config.parse_actor_config("iris"))
        
    def get_world_scene(self):    
        # file_path = self.server_path + "/Users/jame7700/thesis_learning_navigation/world/world_simple_train.usda"
        file_path = self.server_path + "/Users/jaramy/Thesis-asset/world_simple_train.usda"
        # self.pg.load_environment(file_path)
        # self.pg.load_environment(SIMULATION_ENVIRONMENTS["Default Environment"])
        # omniverse://localhost/NVIDIA/Assets/Isaac/2022.2.1/Isaac/Environments/Grid/default_environment.usd

    def get_target(self):
        radius = 0.1
        color = torch.tensor([0, 1, 0])
        ball = DynamicSphere(
            prim_path=self.default_zero_env_path + "/ball",
            translation=self._ball_position,
            name="target_0",
            radius=radius,
            color=color)
        self._sim_config.apply_articulation_settings("ball", get_prim_at_path(ball.prim_path),
                                                     self._sim_config.parse_actor_config("ball"))
        ball.set_collision_enabled(False)


    def get_observations(self) -> dict:
        self.root_pos, self.root_rot = self._copters.get_world_poses(clone=False)
        # self.pos = self._copters.get_body_masses()
        # self.pos,self.qua = self._copters.get_local_poses()
        # print("position",self.pos,"rot",self.qua)
        self.root_velocities = self._copters.get_velocities(clone=False)

        self.target_pos = self._compute_traj(steps = self.future_traj_steps, step_size=5)
        self.set_targets(self.all_indices)

        # print("target pos0 = ",self.target_pos[:, 1, :2])
        # print("target pos1 = ",self.target_pos[:, 0, :2])

        # print("relative = ", self.target_pos[:, 1, :2] - self.target_pos[:, 0, :2])
        # self.ref_err = normalize(self.target_pos[:, 1, :2] - self.target_pos[:, 0, :2])
        self.ref_err = self.target_pos[:, 1, :2] - self.target_pos[:, 0, :2]
        self.ref_heading = torch.atan2(self.ref_err[:, 1], self.ref_err[:, 0]).unsqueeze(1)  #size [512,1] num_envs
        print("ref_heading = ",self.ref_heading)
        print("ref_heading shape = ",self.ref_heading.shape)  
        # print("target pos = ",self.target_pos)
        # print("target pos shape = ",self.target_pos.shape)
        
        root_positions = self.root_pos - self._env_pos

        # print("root pos = ",root_positions.shape)
        # print("target pos = ",self.target_pos.shape)
        
        rpos = self.target_pos - root_positions.unsqueeze(1)
        # print("rpos = ",rpos)
        # print("rpos shape = ",rpos.shape)

        # root_positions = self.root_pos - self._env_pos
        root_quats = self.root_rot

        # print(root_quats)
        rot_x = quat_axis(root_quats, 0)
        rot_y = quat_axis(root_quats, 1)
        rot_z = quat_axis(root_quats, 2)


        # self.heading_err = self.ref_heading - normalize(rot_x[:, :2])
        # print("ref_heading un shape = ",self.ref_heading.unsqueeze(1).shape)
        # print("ref_heading un = ",self.ref_heading.unsqueeze(1))
        # print("normalize shape  = ",normalize(rot_x[:, :2]).shape)
        # print("normalize  = ",normalize(rot_x[:, :2]))
        # print("heading error = ",self.heading_err)
        # print("headin/g error shape = ",self.heading_err.shape)



        root_linvels = self.root_velocities[:, :3]
        root_angvels = self.root_velocities[:, 3:]
        
 
        self.obs_buf[:, 0:3] = self.target_pos[:,0,:] - root_positions
        self.obs_buf[:, 3:6] = rot_x     #1,0,0
        self.obs_buf[:, 6:9] = rot_y     #0,1,0
        self.obs_buf[:, 9:12] = rot_z    #0,0,1

        self.obs_buf[:, 12:15] = root_linvels
        self.obs_buf[:, 15:18] = root_angvels

        self.obs_buf[:, 18:21] = rpos[:,0,:]
        self.obs_buf[:, 21:24] = rpos[:,1,:]
        self.obs_buf[:, 24:27] = rpos[:,2,:]
        self.obs_buf[:, 27:30] = rpos[:,3,:]
        # self.obs_buf[:, 30:33] = rpos[:,4,:]
        # self.prev_obs_buf[..., 18:36] = self.obs_buf
        # self.obs_buf[..., 18:36] = self.prev_obs_buf
        # self.obs_buf[..., 36:40] = self.prev_actions
        # print("prev actions = ",self.prev_actions)

        # self.obs_buf[:, 18:22] = self.prev_actions  #still emtpy


        # print(self.obs_buf)
        # self.error_log.append(torch.sqrt(torch.square(self.target_positions[0] - root_positions[0]).sum(-1)).item())
        # print("error log = ",self.error_log)
        # print("obs_log shape",self.obs_log.shape)
        # print("obs_buf shape",self.obs_buf.shape)
        # print("obs_buf ",self.obs_buf[0,:].shape)

        self.obs_log = torch.cat((self.obs_log,self.obs_buf[None,0,:]),dim=0)  #for logging
        

        observations = {
            self._copters.name: {
                "obs_buf": self.obs_buf
            }
        }
        # print("observations = ",self.obs_buf[:1, 0:3])
        
        return observations
    

    def pre_physics_step(self, actions) -> None:

        if not self._env._world.is_playing():
            return
        
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        

        set_target_ids = (self.progress_buf % 500 == 0).nonzero(as_tuple=False).squeeze(-1)
        # if len(set_target_ids) > 0:
        #     self.set_targets(set_target_ids)

            
        actions = actions.clone().to(self._device)
        self.actions = actions

        # actions = torch.randn((self._num_envs, 4), device=self._device, dtype=torch.float32)
        # actions = actions * -1 

        # actions = torch.tensor([[0.0,0.0,0.0,0.1],
        #                         [0.0,0.0,0.0,0.2],
        #                         [0.0,0.0,0.0,-0.3]], device=self._device, dtype=torch.float32) #for testing
        # actions = torch.tensor([[0.3,0.0,0.0,0.0],[0.0,0.0,0.0,0.2]], device=self._device, dtype=torch.float32) #for testing

        # print("actions = ",actions.shape)
        # print(actions)
        # clamp to [-1.0, 1.0]


        # thrust_cmd_noise = 0.01 * torch.randn(1, dtype=torch.float32, device=self._device)
        # thrust_cmd_noise =  actions[:,0] * thrust_cmd_noise
        # actions[:,0] = actions[:,0] + thrust_cmd_noise

        # rot_cmd_noise = 0.01 * torch.randn(3, dtype=torch.float32, device=self._device)
        # rot_cmd_noise =  actions[:,1:] * rot_cmd_noise
        # actions[:,1:] = actions[:,1:] + rot_cmd_noise

        self.actions_cmds[:,0] = torch.clamp(actions[:,0], min=0.0, max=1.0)
        self.actions_cmds[:,1:] = torch.clamp(actions[:,1:], min=-1.0, max=1.0)


        # self.actions_cmds[:,1:] = torch.zeros_like(actions[:,1:])
        
        self.actions_reward = self.actions_cmds

        thrust_max = self.grav_z * self.mass * self.thrust_to_weight
        self.thrust_max = torch.tensor(thrust_max, device=self._device, dtype=torch.float32)
        # print("thrust_max = ",self.thrust_max)
        # print("thrust_max = ",self.thrust_max.shape)

        self.k_roll = 10.0
        self.k_pitch = 10.0 
        self.k_yaw = 10.0

        self.actions_cmds[:,0] = self.actions_cmds[:,0]* self.thrust_max
        self.actions_cmds[:,1] = self.actions_cmds[:,1]* self.k_roll
        self.actions_cmds[:,2] = self.actions_cmds[:,2]* self.k_pitch
        self.actions_cmds[:,3] = self.actions_cmds[:,3]* self.k_yaw


        # self.csv_writer.writerow([self.actions_cmds[:,0].item(), self.actions_cmds[:,1].item(), self.actions_cmds[:,2].item(), self.actions_cmds[:,3].item()])
        # print("actions = ",actions_cmds[:,0])
        
        # self.actions_cmds_pd = pd.Series(self.actions_cmds.cpu(), index=self.log_action_df.columns)
        # self.log_action_df = self.log_action_df.append(self.actions_cmds_pd, ignore_index=True)
        

        # scale to [0.0, 1.0]
        # thrust_cmds = (thrust_cmds + 1.0) / 2.0
        # print("actions commands [thrust,r,p,y] = \n",actions_cmds)
        # print("actions commands [thrust,r,p,y] size  = \n",actions_cmds.shape)

        # rb = self.world.dc_interface.get_rigid_body("/World/envs/env_0/quadrotor/body")
        # print("rb = ",rb)
        # rotors = [self.world.dc_interface.get_rigid_body("/World/envs/env_0/quadrotor/rotor" + str(i)) for i in range(self._num_rotors)]
        # print("rotor = ",rotors)

        # relative_poses = self.world.dc_interface.get_relative_body_poses(rb, rotors)
        # print("relative_poses = ",relative_poses)
        # print("relative_poses = ",relative_poses[2].p[1])

        aloc_matrix = torch.zeros((4, self._num_rotors), device=self._device, dtype=torch.float32)
        aloc_matrix[0, :] = torch.tensor(self._rotor_constant, device=self._device, dtype=torch.float32) 
        aloc_matrix[1, :] = torch.tensor([self.relative_poses_y[i] * self._rotor_constant[i] for i in range(self._num_rotors)], device=self._device, dtype=torch.float32)
        aloc_matrix[2, :] = torch.tensor([-1*self.relative_poses_x[i] * self._rotor_constant[i] for i in range(self._num_rotors)], device=self._device, dtype=torch.float32)
        aloc_matrix[3, :] = torch.tensor([self._rolling_moment_coefficient[i] * self._rot_dir[i] for i in range(self._num_rotors)], device=self._device, dtype=torch.float32)

        # print("aloc_matrix \n",aloc_matrix)
        # aloc_matrix = torch.squeeze(aloc_matrix)
        # force = 20.0
        # torque = torch.tensor([0.02,0,0])

        aloc_matrix_inv = torch.pinverse(aloc_matrix)
        aloc_matrix_inv = aloc_matrix_inv.to(self._device)
        # print("aloc_matrix inv \n",aloc_matrix_inv)
        
        
        # print("shape aloc = ",aloc_matrix_inv.shape)
        # print("action = ",actions.shape)
        # print(aloc_matrix.shape)
        
  
        # print("actions =",actions)

        actions_T = torch.transpose(self.actions_cmds, 0, 1)
        # print("actions  = ",thrust_cmds)
        # print("actions transpose = ",actions_T)

        # squared_ang_vel = aloc_matrix_inv @ torch.tensor([[10.0, 0.10, 0.0, 0.0]], device=self._device, dtype=torch.float32)
        squared_ang_vel = aloc_matrix_inv @ actions_T
        
        # squared_ang_vel =  torch.matmul(aloc_matrix_inv, actions_T)  
        # print("squared_ang_vel \n ",squared_ang_vel)

        # print(squared_ang_vel)
        squared_ang_vel[squared_ang_vel < 0] = 0.0
        # print(squared_ang_vel)

        max_thrust_vel_squared = torch.pow(torch.tensor(self.max_rotor_velocity[0]), 2.0)
        max_val = torch.max(squared_ang_vel)

        if max_val >= max_thrust_vel_squared:
            normalize = torch.maximum(max_val / max_thrust_vel_squared, torch.tensor(1.0))
            squared_ang_vel = squared_ang_vel / normalize

        ang_vel = torch.sqrt(squared_ang_vel)
    
        # print(ang_vel)
        # print(ang_vel.shape)

        ang_vel_T = torch.transpose(ang_vel, 0, 1)
        # print("ang_vel = \n",ang_vel_T)
        # print(ang_vel_T.shape)

        self.velocity = torch.clamp(ang_vel_T, min=0.0, max=1100.0)
        # print("velocity \n",self.velocity)
        # print("rotor_constant = ",self.thrust_curve._rotor_constant)
        self.rotor_constant = torch.tensor(self._rotor_constant, device=self._device, dtype=torch.float32)
        # print("rotor_constant = ",self.rotor_constant.shape)
        # self.rotor_constant = torch.transpose(self.rotor_constant, 1, 0)

        self.force = self.rotor_constant * torch.pow(self.velocity, 2.0)
        # self.force = torch.tensor([[0.2,0.2,0.2,0.2]], device=self._device, dtype=torch.float32)  ##test
        # print("force = ",self.force)

        self.rolling_moment = 0.0
        self.rolling_moment_coefficient = torch.tensor(self._rolling_moment_coefficient, device=self._device, dtype=torch.float32) 
        self.rot_dir = torch.tensor(self._rot_dir, device=self._device, dtype=torch.float32)   
        self.rolling_moment = self.rolling_moment_coefficient * self.rot_dir * torch.pow(self.velocity, 2.0)

        # print("rolling moment shape ",self.rolling_moment.shape)
        # print("rolling moment ",self.rolling_moment)
        
        self.sum_rolling_moment = torch.sum(self.rolling_moment,1 ,keepdim=True)
        
        # print("sum rolling moment shape ",self.sum_rolling_moment.shape)
        # print("sum rolling moment ",self.sum_rolling_moment)
        # self.sum_rolling_moment = -1.0
        
        # print("rolling ",self.rolling_moment)
        # print("sum of rolling ",torch.sum(self.rolling_moment))
        # print(self.rolling_moment.shape)
        # self.force = self.force.to("cpu")
        
        
        self.force = self.force.reshape(-1, 4, 1)
        # print( "force reshape = \n",self.force)
        # print( "force reshape = \n",self.force.shape)


        force_x = torch.zeros(self._num_envs, 4, dtype=torch.float32, device=self._device)
        force_y = torch.zeros(self._num_envs, 4, dtype=torch.float32, device=self._device)
        force_xy = torch.cat((force_x, force_y), 1)
        force_xy = force_xy.reshape(-1, 4, 2)
        
        
        self.thrusts = torch.cat((force_xy, self.force), 2)

        # self.motor_assymetry = np.array([1.0, 1.0, 1.0, 1.0])
        # re-normalizing to sum-up to 4
        # self.motor_assymetry = self.motor_assymetry * 4. / np.sum(self.motor_assymetry)

        # print("thrust  = ",thrusts)
        
        # thrusts = thrusts * self.thrust_max

        # root_quats = self.root_rot
        # rot_x = quat_axis(root_quats, 0)
        # rot_y = quat_axis(root_quats, 1)
        # rot_z = quat_axis(root_quats, 2)
        # rot_matrix = torch.cat((rot_x, rot_y, rot_z), 1).reshape(-1, 3, 3)

        # print("rot_matrix = \n",rot_matrix)
        # thrusts_0 = thrusts[:, 0]
        # thrusts_0 = thrusts_0[:, :, None]

        # thrusts_1 = thrusts[:, 1]
        # thrusts_1 = thrusts_1[:, :, None]

        # thrusts_2 = thrusts[:, 2]
        # thrusts_2 = thrusts_2[:, :, None]

        # thrusts_3 = thrusts[:, 3]
        # thrusts_3 = thrusts_3[:, :, None]

        # mod_thrusts_0 = torch.matmul(rot_matrix, thrusts_0)
        # mod_thrusts_1 = torch.matmul(rot_matrix, thrusts_1)
        # mod_thrusts_2 = torch.matmul(rot_matrix, thrusts_2)
        # mod_thrusts_3 = torch.matmul(rot_matrix, thrusts_3)

        # self.thrusts[:, 0] = torch.squeeze(mod_thrusts_0)
        # self.thrusts[:, 1] = torch.squeeze(mod_thrusts_1)
        # self.thrusts[:, 2] = torch.squeeze(mod_thrusts_2)
        # self.thrusts[:, 3] = torch.squeeze(mod_thrusts_3)


        # print(thrusts)
        # print("thrusts = \n",thrusts)

        # self.force = torch.tensor([[0,0,1,0]], device=self._device, dtype=torch.float32)
        prop_rot = self.force * self.prop_max_rot
        
        # print("prop_rot = \n",prop_rot)
        # print("prop_rot = \n",prop_rot.shape)

        prop_rot = torch.squeeze(prop_rot, -1)
        # prop_rot =     prop_rot.squeeze()
        # print("prop_rot = \n",prop_rot)
        # print("prop_rot = \n",prop_rot.shape)
        
        # print("1st",prop_rot[:, 0])
        self.dof_vel[:, 0] = prop_rot[:, 0]
        self.dof_vel[:, 1] = -1.0 * prop_rot[:, 1]
        self.dof_vel[:, 2] = prop_rot[:, 2]
        self.dof_vel[:, 3] = -1.0 * prop_rot[:, 3]

        # print("propeller vel = \n",self.dof_vel)
        self._copters.set_joint_velocities(self.dof_vel)
        # print("thrust \n ",thrusts)
        for i in range(4):
            # print("force = ",self.force[:, i])
            # pass
            self._copters.physics_rotors[i].apply_forces_and_torques_at_pos(forces = self.thrusts[:,i], indices=self.all_indices, is_global=False)
            # self._copters.physics_rotors[i].apply_forces([[[0.0,0.0,0.2]]], indices=self.all_indices,is_global=False)
        
        # from omni.isaac.dynamic_control import _dynamic_control
        # dc = _dynamic_control.acquire_dynamic_control_interface()
        
        # self.sum_rolling_moment = self.sum_rolling_moment*1.0
        # print("sum of rolling =\n",self.sum_rolling_moment)
        torque_x = torch.zeros(self._num_envs, 1, dtype=torch.float32, device=self._device)
        torque_y = torch.zeros(self._num_envs, 1, dtype=torch.float32, device=self._device)
        torque_xy = torch.cat((torque_x, torque_y), 1)
        # torque_xy = torque_xy.reshape(-1, 2, 2)
        # print("torque_xy",torque_xy)
        # print("txy shape",torque_xy.shape)
        # print("\n sum ",self.sum_rolling_moment)
        # print("sum shape",self.sum_rolling_moment.shape)
        self.torques = torch.cat((torque_xy, self.sum_rolling_moment ), 1)
        # print("torque = \n",self.torques)
        # print("torque = \n",self.torques.shape)


        # torch_rol = torch.tensor([[0.0,0.0,self.sum_rolling_moment],[0.0,0.0,self.sum_rolling_moment]], dtype=torch.float32, device=self._device)
        # print("torch_rol",torch_rol)
        # self._copters.physics_bodys.apply_forces_and_torques_at_pos(torques = torch_rol, indices=self.all_indices, is_global=False)
        self._copters.physics_bodys.apply_forces_and_torques_at_pos(torques = self.torques.reshape(-1, 3), indices=self.all_indices, is_global=False)
        
        # rb = self.world.dc_interface.get_rigid_body("/World/envs/env_0/quadrotor" + "/body")        
        # Apply the torque to the rigidbody. The torque should be expressed in the rigidbody frame
        # self.world.dc_interface.apply_body_torque(rb, carb._carb.Float3([0.0,0.0,self.sum_rolling_moment]), False)
        # print("mass ",self._copters.physics_bodys.get_masses(indices=self.all_indices))
        # apply_(self.rolling_moment, indices=self.all_indices,is_global=False)

    def post_reset(self):

        self.actions = torch.zeros((self._num_envs, 4), device=self._device, dtype=torch.float32)
        self.all_indices = torch.arange(self._num_envs, dtype=torch.int32, device=self._device)
    
        thrust_max = self.grav_z * self.mass * self.thrust_to_weight
        
        self.thrust_max = torch.tensor(thrust_max, device=self._device, dtype=torch.float32)
        self.actions_cmds = torch.zeros((self._num_envs, 4), device=self._device, dtype=torch.float32)
        
        # self.target_positions = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        # self.target_positions[:, 2] = 1  

        self.root_pos, self.root_rot = self._copters.get_world_poses()
        self.root_velocities = self._copters.get_velocities()
        self.dof_pos = self._copters.get_joint_positions()
        self.dof_vel = self._copters.get_joint_velocities()

        self.initial_ball_pos, self.initial_ball_rot = self._balls.get_world_poses(clone=False)
        self.initial_root_pos, self.initial_root_rot = self.root_pos.clone(), self.root_rot.clone()

        # control parameters
        self.thrusts = torch.zeros((self._num_envs, 4, 3), dtype=torch.float32, device=self._device)
        self.torques = torch.zeros((self._num_envs, 3), dtype=torch.float32, device=self._device)
    
        # self.set_targets(self.all_indices)

    def set_targets(self, env_ids):
        # num_sets = len(env_ids)
        envs_long = env_ids.long()
        # set target position randomly with x, y in (-1, 1) and z in (1, 2)
        # self.target_positions[envs_long, 0:2] = torch.rand((num_sets, 2), device=self._device) * 2 - 1
        # self.target_positions[envs_long, 2] = torch.rand(num_sets, device=self._device) + 1
        ##_____
        # self.target_positions[envs_long, 0:2] = torch.ones((num_sets, 2), device=self._device)*0.7
        # self.target_positions[envs_long, 2] = torch.ones(num_sets, device=self._device) * 1.0
        ##_____
        
        #self.target_positions[envs_long, 0:2] = torch.rand((num_sets, 2), device=self._device)
        # self.target_positions[envs_long, 2] = torch.rand(num_sets, device=self._device)*0.15 + 2.0

        # shift the target up so it visually aligns better self.target_positions[envs_long]
        ball_pos = self.target_pos[:,0,:] + self._env_pos[envs_long]
        # ball_pos[:, 2] += 0.0
        self._balls.set_world_poses(ball_pos[:, 0:3])
    
    def plot_guide_path(self) :
        # t = torch.arange(0, 2*torch.pi, 2.0, device=self._device)
        # y = torch.sin(t)
        # x = t 
        # # z = torch.tensor([1.0], dtype=torch.float32)
        # z = torch.ones_like(t)

        # traj_vis = torch.stack((x, y, z), dim=1)
        # traj_vis = traj_vis + self._env_pos[0]
        traj_vis = self._compute_traj(steps = 300, env_ids = self.central_env_idx)
        traj_vis = traj_vis + self._env_pos[self.central_env_idx]
        
        point_list_0 = traj_vis[:-1].tolist()
        point_list_1 = traj_vis[1:].tolist()

        colors = [(1.0, 0.0, 0.0, 1.0) for _ in range(len(point_list_0))]
        sizes = [1 for _ in range(len(point_list_0))]
        self.draw.draw_lines(point_list_0, point_list_1, colors, sizes)
    
    def scale_time(self, t, a: float=1.0):
        return t / (1 + 1/(a*torch.abs(t)))
    
    def sin_traj(self, t, c: float=1.0):
        sin_t = torch.sin(t)
        y = c*t
        z = torch.ones_like(t)
        pose_traj = torch.stack([y, sin_t, z], dim=-1)
        return pose_traj
    
    def _compute_traj(self, steps: int, env_ids=None, step_size: float=1.):
        if env_ids is None:
            env_ids = ...

        # print(self.progress_buf[env_ids].shape)
        # print(self.progress_buf[env_ids])
        # t = self.progress_buf[env_ids].unsqueeze(1) + step_size * torch.arange(steps, device=self.device)
        # print(self.progress_buf[env_ids])
        t = self.progress_buf[env_ids].unsqueeze(-1) + step_size * torch.arange(steps, device=self.device)
        # print("t ",t.shape)
        # print(t)
        # t = self.scale_time(self.traj_w[env_ids] * 2*t * self.dt)
        t = self.scale_time(2 * t * self.dt)

        # print("t = ", t)
        # print("t shape = ", t.shape)

        traj_target = self.sin_traj(t, c=1.0)
        
        # print("shape target_pos",self.target_pos.shape)
        return self.origin + traj_target
       
    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        self.dof_pos[env_ids, :] = torch_rand_float(-0.0, 0.0, (num_resets, self._copters.num_dof), device=self._device)
        self.dof_vel[env_ids, :] = 0

        root_pos = self.initial_root_pos.clone()
        root_pos[env_ids, 0] += torch_rand_float(-0.0, 0.0, (num_resets, 1), device=self._device).view(-1)
        root_pos[env_ids, 1] += torch_rand_float(-0.0, 0.0, (num_resets, 1), device=self._device).view(-1)
        root_pos[env_ids, 2] += torch_rand_float(-0.0, 0.0, (num_resets, 1), device=self._device).view(-1)
        root_velocities = self.root_velocities.clone()
        root_velocities[env_ids] = 0

        # apply resets
        self._copters.set_joint_positions(self.dof_pos[env_ids], indices=env_ids)
        self._copters.set_joint_velocities(self.dof_vel[env_ids], indices=env_ids)

        self._copters.set_world_poses(root_pos[env_ids], self.initial_root_rot[env_ids].clone(), indices=env_ids)
        self._copters.set_velocities(root_velocities[env_ids], indices=env_ids)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        self.thrusts[env_ids] = 0
        self.torques[env_ids] = 0
        # self.thrust_cmds_damp[env_ids] = 0
        # self.thrust_rot_damp[env_ids] = 0
        
        if self.render:
            self.plot_guide_path()
            self.render = False
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"][key] = torch.mean(
                self.episode_sums[key][env_ids]) / self._max_episode_length
            self.episode_sums[key][env_ids] = 0.0

    def calculate_metrics(self) -> None:
        root_positions = self.root_pos - self._env_pos
        # print("root pos ",root_positions)
        
        root_quats = self.root_rot
        root_angvels = self.root_velocities[:, 3:]

        # pos reward
        target_dist = torch.sqrt(torch.square(self.target_pos[:,0,:] - root_positions).sum(-1))
        # print("env 0 target dist = ",target_dist[0])
        pos_reward = torch.exp(-2.0 * (1*target_dist))
        
        # pos_reward = 1.0 / (1.0 + target_dist)
        # print("pos reward =\n",pos_reward)
        
        ones_reward = torch.ones_like(target_dist)
        zero_reward = torch.zeros_like(target_dist)

        extra_pos_reward = torch.where(target_dist < 0.04, ones_reward*2.0, zero_reward)
        pos_penalty = torch.where(target_dist > 0.35, (ones_reward*target_dist)*0.01, zero_reward)

        self.target_dist = target_dist
        self.root_positions = root_positions

        # orient reward
        ups = quat_axis(root_quats, 2)
        # print("ups \n", ups)

        self.orient_z = ups[:, 2]
        # up_reward = torch.clamp(ups[:, 2], min=0.0, max=0.5)
        up_reward = 0.5 / (1.0 + torch.square(self.orient_z))
        # print("up reward \n", up_reward)
        # ups = quat_axis(root_quats, 2)
        

        # effort reward
        # effort = torch.square(self.actions).sum(-1)  #using thrust commands instead of actions  
        effort = self.actions_reward[:,0]
        effort_reward = 0.5 * torch.exp(-0.5 * effort)

        # spin reward
        spin = torch.square(root_angvels).sum(-1)
        spin_reward = 0.3 * torch.exp(-1.0 * spin)
        spin_penalty = torch.where(spin > 1.0, ones_reward * 0.1, zero_reward)
        
        # spinnage = torch.abs(root_angvels[..., 2])
        # spinnage_reward = 1.0 / (1.0 + 10 * spinnage * spinnage)

        # combined reward
        #self.rew_buf[:] = pos_reward + pos_reward * (up_reward + spin_reward) - effort_reward
        self.rew_buf[:] = 1.5*pos_reward + (1.5*pos_reward * (up_reward)) + (pos_reward * (spin_reward)) - spin_penalty 
        
        # print("pos_reward = ",self.rew_buf)
        # self.rew_buf[:] = pos_reward + pos_reward * (up_reward + spin_reward) 


        # log episode reward sums
        self.episode_sums["rew_pos"] += pos_reward
        self.episode_sums["rew_orient"] += up_reward
        self.episode_sums["rew_effort"] += effort_reward
        self.episode_sums["rew_spin"] += spin_reward #spin_reward

        # log raw info
        self.episode_sums["raw_dist"] += target_dist
        self.episode_sums["raw_orient"] += ups[..., 2]
        self.episode_sums["raw_effort"] += effort
        self.episode_sums["raw_spin"] += spin #spin

        # print(self.episode_sums)

    def is_done(self) -> None:
        # resets due to misbehavior
        ones = torch.ones_like(self.reset_buf)
        die = torch.zeros_like(self.reset_buf)

        # print("target_dist = ",self.target_dist)
        die = torch.where(self.target_dist > 0.5, ones, die)

        # z >= 0.5 & z <= 5.0 & up > 0
        # print("root pos = ",self.root_positions[:, 2])
        die = torch.where(self.root_positions[..., 2] < 0.3, ones, die)

        # print("die = ",die)

        # die = torch.where(self.root_positions[..., 2] > 10.0, ones, die)
        # print("die2 = ",die)
        
        die = torch.where(self.orient_z < 0.0, ones, die)
        # print("die3 = ",die)

        # resets due to episode length
        self.reset_buf[:] = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, die)



