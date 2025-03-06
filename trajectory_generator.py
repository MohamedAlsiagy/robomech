import itertools
import math
import pandas as pd
import numpy as np

robots_folder = "/workspace/Inshallah/Robots"
robot_name = "robot_0"
urdf_path = f"/workspace/Inshallah/corrected_inertias_tests/robot.urdf"
usd_path = f"/workspace/Inshallah/corrected_inertias_tests/robot.usd"
STAGE_PATH = "/" + robot_name
time_per_perm = 4
number_of_waves_per_perm = 3
initial_amplitude = 3

axis_equivelance = 1.5
axis_correction = [1 , axis_equivelance , axis_equivelance , 1 , axis_equivelance , 1]
Amplitude_correcction = [axis_correction[i] * initial_amplitude/2**math.sqrt(i) for i in range(6)]

def sin_pro_max(t, f_req):
    tanh_term = np.tanh(f_req * t)**2
    sin_term = np.sin(f_req * t)**2
    return (tanh_term + (1 - tanh_term) * sin_term) * np.sin(f_req * t)

# Define the sine function
def sin_wave(t):
    f = (2 * math.pi * number_of_waves_per_perm)/(time_per_perm)
    return sin_pro_max(t , f)

# Define the zero function
def zero_wave(t):
    return 0

# Create a list with 6 sine waves and 6 zeros
elements = [sin_wave] * 6 + [zero_wave] * 6

# Generate all unique permutations
unique_perms = set(itertools.permutations(elements, 6))
permutation_functions = [list(perm) for perm in unique_perms]

import sys
import os
import numpy as np
from isaacsim import SimulationApp

CONFIG = {"renderer": "RayTracedLighting", "headless": False}

# Example ROS2 bridge sample demonstrating the manual loading of stages
# and creation of ROS components
simulation_app = SimulationApp(CONFIG)
import carb
import omni.kit.app
import omni.graph.core as og
import usdrt.Sdf
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core import SimulationContext
from omni.isaac.lab.sim.converters import UrdfConverter, UrdfConverterCfg
from omni.isaac.lab.utils.assets import check_file_path
from omni.isaac.core.utils import extensions, prims, rotations, stage, viewports
from omni.isaac.nucleus import get_assets_root_path
from pxr import Gf
from omni.isaac.core.utils.prims import define_prim , get_all_matching_child_prims , get_prim_path , get_prim_type_name
import omni.isaac.core.utils.prims as prim_utils
import pinocchio as pin
import numpy as np
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.core.utils.prims import get_all_matching_child_prims , get_prim_type_name
from omni.isaac.sensor.scripts.effort_sensor import EffortSensor
from math import pi
import time 

# Open the new stage (specified by urdf_converter.usd_path)
stage_utils.open_stage(usd_path)
stage = omni.usd.get_context().get_stage()

model = pin.buildModelFromUrdf(urdf_path)
gravity_data = model.createData()

# Define gravity vector (default is [0, 0, -9.81] for z-up)
gravity = np.array([0, 0, -9.81])  # Adjust if your URDF uses a different convention

# Set the gravity in the model
model.gravity.linear = gravity

# Get all joints under ArticulationRoot
joint_prims = get_all_matching_child_prims(
    prim_path=STAGE_PATH, predicate=lambda a: "RevoluteJoint" in get_prim_type_name(a)
)

effort_sensors = []

# Modify joint properties
for joint_prim in joint_prims:
    joint_prim.GetAttribute("drive:angular:physics:maxForce").Set(100000.0)
    joint_prim.GetAttribute("drive:angular:physics:damping").Set(0.0)
    joint_prim.GetAttribute("drive:angular:physics:stiffness").Set(0.0)
    joint_prim.GetAttribute("physics:lowerLimit").Set(-np.inf)
    joint_prim.GetAttribute("physics:upperLimit").Set(np.inf)

    sensor = EffortSensor(
        prim_path=str(joint_prim.GetPath()),
        sensor_period= 0.005, #~200 Hz
        use_latest_data=False,
        enabled=True)
    
    effort_sensors.append(sensor)

# Check if the stage is loaded
simulation_context = SimulationContext(stage_units_in_meters=1.0)

# Check the physics context
simulation_context.initialize_physics()

# need to initialize physics getting any articulation..etc
simulation_context.play()

# Initialize Omniverse Dynamic Control
dc = _dynamic_control.acquire_dynamic_control_interface()
art = dc.get_articulation(f"{STAGE_PATH}/link_0") 
dc.wake_up_articulation(art)

tau_practical = [0 for i in range(len(joint_prims))]
tau_gravity = np.zeros(len(joint_prims))
tau = [0 for i in range(len(joint_prims))]

start_time = time.time()
perm_index = 0
trajectory_function_list = permutation_functions[perm_index]

last_joint_velocities = np.zeros(len(joint_prims))
last_t = start_time

print(f"testing on {len(permutation_functions)} permutations")
print(f"currently on permutation : {perm_index}")
print([traj.__name__ for traj in trajectory_function_list])
mode = "".join(["1" if traj.__name__ == "sin_wave" else "0" for traj in trajectory_function_list])

data = {
    "time": [],
    "mode" : [],
    "Joint": [],
    "Position": [],
    "Velocity": [],
    "Acceleration": [],
    "Target_Torque_GC": [],
    "Effort_GC": [],
    "Target_Torque": [],
    "Effort": [],
}

while simulation_app.is_running():  
    # Run with a fixed step size
    simulation_context.step(render=True)

    joint_positions = np.zeros(len(joint_prims))
    joint_velocities = np.zeros(len(joint_prims))
    effort_readings = np.zeros(len(joint_prims))

    for i , joint_prim in enumerate(joint_prims):
        joint_positions[i] = joint_prim.GetAttribute("state:angular:physics:position").Get()
        joint_velocities[i] = joint_prim.GetAttribute("state:angular:physics:velocity").Get()
        effort_readings[i] = sensor.get_sensor_reading(use_latest_data = True).value
    
    current_t = time.time()
    dt = (current_t - last_t)
    joint_accelerations = (joint_velocities - last_joint_velocities) / dt
    last_t = current_t
    last_joint_velocities = joint_velocities.copy()

    for j in range(len(joint_prims)):
        data["time"].append(current_t - start_time)
        data["mode"].append(mode)
        data["Joint"].append(j)
        data["Position"].append(joint_positions[j])
        data["Velocity"].append(joint_velocities[j])
        data["Acceleration"].append(joint_accelerations[j])
        data["Target_Torque_GC"].append(tau_practical[j])
        data["Effort_GC"].append(effort_readings[j] - tau_gravity[j])
        data["Target_Torque"].append(tau[j])
        data["Effort"].append(effort_readings[j])

    # print("Joint positions (angles):", joint_positions)

    # Define the robot's configuration (joint positions) from Isaac Sim
    q = (np.array(joint_positions) + (np.array(joint_velocities) * dt/2) + (np.array(joint_accelerations) * (dt**2)/8))  * pi/180 # Use the joint angles read from Isaac Sim
    # Define the robot's velocity and acceleration (zero for gravity compensation)
    v = np.zeros(model.nv)  # Zero velocity
    a = np.zeros(model.nv)  # Zero acceleration

    # Compute gravity compensation torques using RNEA
    tau_gravity = pin.rnea(model, gravity_data, q, v, a)
    # print("Gravity compensation torques:", tau_gravity)
    tau_practical = np.array([Amplitude_correcction[i] * traj_i(time.time() - start_time) for i , traj_i in enumerate(trajectory_function_list)])
    tau =  tau_gravity + tau_practical

    # Apply the computed gravity compensation torques to the robot in Isaac Sim
    dc.set_articulation_dof_efforts(art, tau.tolist())

    if current_t - start_time >= time_per_perm:
        start_time = current_t
        perm_index += 1
        if perm_index == len(permutation_functions):
            break

        trajectory_function_list = permutation_functions[perm_index]

        print("-_" * 8 + "-")
        print(f"currently at permutation : {perm_index}")
        print([traj.__name__ for traj in trajectory_function_list])

        mode = "".join(["1" if traj.__name__ == "sin_wave" else "0" for traj in trajectory_function_list])

        simulation_context.stop()
        simulation_context.play()

trajectory_folder = os.path.join(robots_folder , "trajectories")
if not os.path.exists(trajectory_folder):
    os.makedirs(trajectory_folder)

# Save to CSV
df = pd.DataFrame(data)
df.to_csv(os.path.join(trajectory_folder , f"{STAGE_PATH.replace('/' , '')}_trajectory.csv" ), index=False)

del data

simulation_context.stop()
simulation_app.close()