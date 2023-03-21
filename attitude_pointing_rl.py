#!/usr/bin/env python
# coding: utf-8

from Basilisk.utilities import (
    SimulationBaseClass,
    macros,
    unitTestSupport,
    simulationArchTypes
)

from Basilisk.simulation import (
    spacecraft,
    extForceTorque,
    simpleNav
)

from Basilisk.fswAlgorithms import (
    inertial3D,
    attTrackingError
)

from Basilisk.architecture import messaging

import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import os
import gymnasium as gym
import ray
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch, try_import_tf
from ray.tune.registry import get_trainable_cls
from ray import air, tune
from ray.rllib.utils.test_utils import check_learning_achieved
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--num_rollout_workers",
                    type=int,
                    required=True,
                    help="Number of rollout workers passed to Ray trainer config")
parser.add_argument("--num_gpus",
                    type=int,
                    default=0,
                    help="Number of GPUS passed to Ray trainer config")
parser.add_argument("--stopper_patience",
                    type=int,
                    default=10,
                    help="Patience value to pass to Ray PlateauStopper object")

args = parser.parse_args()

# Start ray. The dashboard host is set to be the rostration server
# accessable on the local SpaceTREx WiFi network in Drake
ray.init()


class Simulation:
    def __init__(self, tumble, desired_orientation, sim_max_time, sim_dt, record=True, num_log_points = 150):
        self.sim_task_name = "simTask"
        self.sim_proc_name = "simProc"
        
        self.record = record
        
        self.sim = SimulationBaseClass.SimBaseClass()
        self.sim_max_time = sim_max_time
        self.sim_dt = sim_dt
        
        self.dyn_process = self.sim.CreateNewProcess(self.sim_proc_name, 10)
        self.dyn_process.addTask(self.sim.CreateNewTask(self.sim_task_name, self.sim_dt))
        
        
        # Setup the spaecraft model.
        # The spacecraft model's documentation is found at
        # http://hanspeterschaub.info/basilisk/Documentation/simulation/dynamics/spacecraft/spacecraft.html
        self.spacecraft = spacecraft.Spacecraft()
        self.spacecraft.ModelTag = "bsk-Sat"
        
        # Define the inertial properties
        self.I = [900., 0., 0.,
                  0., 800., 0.,
                  0., 0., 600.]
        
        self.spacecraft.hub.mHub = 750.0  # spacecraft mass [kg]
        self.spacecraft.hub.r_BcB_B = [[0.0], [0.0], [0.0]]  # m - position vector of body-fixed point B relative to CM
        self.spacecraft.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(self.I)
        self.spacecraft.hub.sigma_BNInit = [[0.1], [0.2], [-0.3]]  # sigma_BN_B
        self.spacecraft.hub.omega_BN_BInit = tumble  # [rad/s]
        
        # Add the spacecraft object to the simulation process
        self.sim.AddModelToTask(self.sim_task_name, self.spacecraft)
        
        # Setup the external control torque
        self.ex_torque = extForceTorque.ExtForceTorque()
        self.ex_torque.ModelTag = "externalDisturbance"
        self.spacecraft.addDynamicEffector(self.ex_torque)
        self.sim.AddModelToTask(self.sim_task_name, self.ex_torque)
        
        # Setup the navigation sensor module which controls the
        # craft's attitude, rate, and position
        self.nav = simpleNav.SimpleNav()
        self.nav.ModelTag = "simpleNavigation"
        self.sim.AddModelToTask(self.sim_task_name, self.nav)
        
        # Setup the inertial 3D guidance module
        self._i3D = inertial3D.inertial3DConfig()
        self.i3D = self.sim.setModelDataWrap(self._i3D)
        self.i3D.ModelTag = "inertial3D"
        self.sim.AddModelToTask(self.sim_task_name, self.i3D, self._i3D)
        self._i3D.sigma_R0N = desired_orientation
        
        # Setup the attitude tracking error evaluation module
        self._attErr = attTrackingError.attTrackingErrorConfig()
        self.attErr = self.sim.setModelDataWrap(self._attErr)
        self.attErr.ModelTag = "attErrorInertial3D"
        self.sim.AddModelToTask(self.sim_task_name, self.attErr, self._attErr)
        
        # Set up recording of values *before* the simulation is initialized
        if self.record:
            t = unitTestSupport.samplingTime(self.sim_max_time, self.sim_dt, num_log_points)
            self.attitude_err_log = self._attErr.attGuidOutMsg.recorder(t)
            self.sim.AddModelToTask(self.sim_task_name, self.attitude_err_log)
        
        # Set up the messaging
        self.nav.scStateInMsg.subscribeTo(self.spacecraft.scStateOutMsg)
        self._attErr.attNavInMsg.subscribeTo(self.nav.attOutMsg)
        self._attErr.attRefInMsg.subscribeTo(self._i3D.attRefOutMsg)
        
    def set_external_torque_cmd_msg(self, msg):
        self.ex_torque.cmdTorqueInMsg.subscribeTo(msg)
        
    def run(self):
        self.sim.InitializeSimulation()
        # self.sim.ConfigureStopTime(self.sim_max_time)
        print(f"Overall stop time is {self.sim_max_time}")
        
        first_leg = self.sim_max_time / 2
        print(f"Simulating up until {first_leg}")
        self.sim.ConfigureStopTime(first_leg)
        self.sim.ExecuteSimulation()
        
        print(f"Simulating now until {self.sim_max_time}")
        self.sim.ConfigureStopTime(self.sim_max_time)
        self.sim.ExecuteSimulation()
        
    def get_plot_data(self):
        if not self.record:
            print("WARNING: Sim did not record!")
            return
        dataLr = self.mrp_log.torqueRequestBody
        dataSigmaBR = self.attitude_err_log.sigma_BR
        dataOmegaBR = self.attitude_err_log.omega_BR_B
        timeAxis = self.attitude_err_log.times()
        
        return dataLr, dataSigmaBR, dataOmegaBR, timeAxis
        
    def plot(dataLr, dataSigmaBR, dataOmegaBR, timeAxis):
        np.set_printoptions(precision=16)

        plt.figure(1)
        for idx in range(3):
            plt.plot(timeAxis * macros.NANO2MIN, dataSigmaBR[:, idx],
                     color=unitTestSupport.getLineColor(idx, 3),
                     label=r'$\sigma_' + str(idx) + '$')
        plt.legend(loc='lower right')
        plt.xlabel('Time [min]')
        plt.ylabel(r'Attitude Error $\sigma_{B/R}$')
        figureList = {}
        pltName = "1"
        figureList[pltName] = plt.figure(1)

        plt.figure(2)
        for idx in range(3):
            plt.plot(timeAxis * macros.NANO2MIN, dataLr[:, idx],
                     color=unitTestSupport.getLineColor(idx, 3),
                     label='$L_{r,' + str(idx) + '}$')
        plt.legend(loc='lower right')
        plt.xlabel('Time [min]')
        plt.ylabel('Control Torque $L_r$ [Nm]')
        pltName = "2" 
        figureList[pltName] = plt.figure(2)

        plt.figure(3)
        for idx in range(3):
            plt.plot(timeAxis * macros.NANO2MIN, dataOmegaBR[:, idx],
                     color=unitTestSupport.getLineColor(idx, 3),
                     label=r'$\omega_{BR,' + str(idx) + '}$')
        plt.legend(loc='lower right')
        plt.xlabel('Time [min]')
        plt.ylabel('Rate Tracking Error [rad/s] ')



# Define Gym environment to train RL-based control of attitude correction
class AttitudeGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvContext):
        self.action_space = gym.spaces.Box(low=-100, high=100, shape=(3,), dtype=float)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=float)

        self.simulation = None
        self.step_size_ns = config['step_size_ns']
        self.max_mission_time_ns = config['max_mission_time_ns']
        self.run_until_ns = None
        self.action_msg = None
        self.record_sim = config['record_sim']
        self.show_debug=config['show_debug']
        self.iter = None

        # self.tumble = [[0.8], [-0.6], [0.5]]
        self.tumble = [[0.001], [-0.01], [0.03]]  # The 'small tumble' from the example [rad/s] - omega_BN_B
        self.desired_ori = [0.0]*3

        self.reset()


    def reset(self, *, seed=None, options=None):
        self.simulation = Simulation(self.tumble, self.desired_ori, self.max_mission_time_ns, self.step_size_ns, record=self.record_sim)
        self.simulation.sim.InitializeSimulation()

        self.obs_space_recorder = self.simulation._attErr.attGuidOutMsg.recorder(self.step_size_ns-1)
        self.simulation.sim.AddModelToTask(self.simulation.sim_task_name, self.obs_space_recorder)

        self.run_until_ns = self.step_size_ns
        self.action_msg = messaging.CmdTorqueBodyMsg()
        self.simulation.set_external_torque_cmd_msg(self.action_msg)
        self.iter = 0

        self._run()

        return np.array(self._get_observation()), {}

    def step(self, action):
        self._debug_msg(f"Iteration {self.iter}")
        msgData = messaging.CmdTorqueBodyMsgPayload()
        msgData.torqueRequestBody = action
        self._debug_msg(f"Publishing torque request = {action}", tab=True)
        self.action_msg.write(msgData)
        self._run()
        
        obs = self._get_observation()
        self._debug_msg(f"Observation is {obs}", tab=True)
        done = self.run_until_ns >= self.max_mission_time_ns
        reward = self._get_reward()
        self._debug_msg(f"Reward is {reward}", tab=True)
        truncated = done
        
        self.iter += 1
        
        return np.array(obs), reward, done, truncated, {}
        
    def render(self):
        raise NotImplementedError
        
    def close(self):
        pass
        
    def _run(self):
        self.simulation.sim.ConfigureStopTime(self.run_until_ns)
        self.simulation.sim.ExecuteSimulation()
        self.run_until_ns += self.step_size_ns
        
    def _get_observation(self):
        sigma_BR_obs = self.obs_space_recorder.sigma_BR[-1]
        omega_BR_B_obs = self.obs_space_recorder.omega_BR_B[-1]
        
        return list(it.chain.from_iterable([sigma_BR_obs, omega_BR_B_obs]))
    
    def _get_reward(self):
        # Reward the agent for getting the positional data closer to 0
        sigma_BR = self.obs_space_recorder.sigma_BR[-1]
        abs_delta = np.linalg.norm(self.desired_ori - sigma_BR)
        # if abs_delta <= 0.1:
        #     self._debug_msg("Delta between current and desired orientaiton is small! Giving positve reward")
        #     reward = 10
        # else:
        #     reward = -10.0 * abs_delta
        reward = -10.0 * abs_delta
        self._debug_msg(f"sigma_BR = {sigma_BR}, desired orientation is {self.desired_ori}, delta is {self.desired_ori - sigma_BR}, and reward is {reward}", tab=True)
        return reward
    
    def _debug_msg(self, msg, tab=False):
        if self.show_debug:
            if tab:
                msg = '\t' + msg
            print(f"[DEBUG] {msg}")


# Define a custom Torch model that just delegates a fully connected net
torch, nn = try_import_torch()
tf1, tf, tfv = try_import_tf()

class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.torch_sub_model = TorchFC(
            obs_space, action_space, num_outputs, model_config, name
        )

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])
    
ModelCatalog.register_custom_model(
    "my_model", TorchCustomModel
)


# Define trainer config for Ray training algorithm
config = (
    get_trainable_cls("PPO")
    .get_default_config()
    # or "corridor" if registered above
    .environment(AttitudeGym, env_config={
        'step_size_ns': macros.sec2nano(0.1),
        'max_mission_time_ns': macros.min2nano(10.0),
        'record_sim': False,
        'show_debug': False
    })
    .framework("torch")
    .rollouts(num_rollout_workers=args.num_rollout_workers, batch_mode="complete_episodes")
    .training(
        # model={
        #     "custom_model": "my_model",
        #     "vf_share_layers": True,
        # },
        # train_batch_size=8000,
    )
    .resources(num_gpus=args.num_gpus)
)

# import pprint
# pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(config.to_dict())

# Define stop conditions
# stop_cond = {
#     "episode_reward_mean" : -500,  # this is somewhat random but based on what the reward would have been in the regular controller
#     "training_iteration": 5000
# }
stop_cond = ray.tune.stopper.ExperimentPlateauStopper(
    metric='episode_reward_mean',
    mode='max',
    patience=args.stopper_patience,  # number of epochs to wait for a change in the model
)


# Perform the training!
tuner = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=air.RunConfig(stop=stop_cond)
)

results = tuner.fit()
# check_learning_achieved(results, stop_cond['episode_reward_mean'])
