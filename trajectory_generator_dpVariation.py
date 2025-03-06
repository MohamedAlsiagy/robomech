import sys
import os
import numpy as np
import itertools
import math
import pandas as pd
import random
from tqdm import tqdm
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
import xml.etree.ElementTree as ET

# Define the path to the robot URDF
robot_urdf_path = "/workspace/Inshallah/corrected_inertias_tests/robot.urdf"
robot_name = "robot_0"
STAGE_PATH = "/" + robot_name
time_per_perm = 4
number_of_waves_per_perm = time_per_perm * 3
initial_amplitude = 9

variability_percentage = 0.5
axis_equivelance = 4
axis_correction = [1 , axis_equivelance , axis_equivelance , 1 , axis_equivelance , 1]
Amplitude_correcction = [axis_correction[i] * initial_amplitude/2**math.sqrt(i) for i in range(6)]

# Viscous and Coulomb friction parameters for a medium-sized robot
max_viscous_friction = 0.05  # Maximum viscous friction coefficient 
min_viscous_friction = 0.001  # Minimum viscous friction coefficient

max_coulomb_friction = 0.5  # Maximum Coulomb friction
min_coulomb_friction = 0.005  # Minimum Coulomb friction 


# Define the sine function
def sin_pro_max(t, f_req):
    tanh_term = np.tanh(f_req * t)**2
    sin_term = np.sin(f_req * t)**2
    return (tanh_term + (1 - tanh_term) * sin_term) * np.sin(f_req * t)

# Define the sine function
def sin_wave(t , random_shift = 0):
    f = (2 * math.pi * number_of_waves_per_perm)/(time_per_perm)
    return sin_pro_max(t + random_shift , f)

# Define the zero function
def zero_wave(t , **kwargs):
    return 0

def predict_position_midstep(x, v, a, dt):
    dt_half = dt / 2
    return x + v * dt_half + 0.5 * a * dt_half**2

# Create a list with 6 sine waves and 6 zeros
elements = [sin_wave] * 6 + [zero_wave] * 6

# Generate all unique permutations
unique_perms = set(itertools.permutations(elements, 6))
permutation_functions = [list(perm) for perm in unique_perms]

instances_dynamic_params = []

# Function to modify URDF dynamics
def modify_urdf_dynamics(urdf_path, output_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    dynamic_params = [[] for i in range(6)]
            
    for joint_idx , joint in enumerate(root.findall("joint")):
        dynamics = joint.find("dynamics")
        if dynamics is not None:
            coloumb_friction = random.uniform(min_coulomb_friction, max_coulomb_friction)
            dynamics.set("damping", str(coloumb_friction))

            viscous_friction = random.uniform(min_viscous_friction, max_viscous_friction)

            dynamic_params[joint_idx].append(coloumb_friction)
            dynamic_params[joint_idx].append(viscous_friction)
            
    for link_idx , link in enumerate(root.findall("link")[1:]):
        inertial = link.find("inertial")
        if inertial is not None:
            inertia = inertial.find("inertia")
            if inertia is not None:
                for attr in ["ixx", "ixy", "ixz", "iyy", "iyz", "izz"]:
                    value = float(inertia.get(attr, 0))
                    value *= random.uniform(1 - variability_percentage, 1 + variability_percentage)  # Random change within 50%
                    inertia.set(attr, str(value))

                    dynamic_params[link_idx].append(value)
    
    instances_dynamic_params.append(dynamic_params)
    tree.write(output_path)

# Create instances folder
instances_folder = os.path.join(os.path.dirname(robot_urdf_path), "instances")
if not os.path.exists(instances_folder):
    os.makedirs(instances_folder)

# Generate instances with modified dynamics
num_instances = 4 * len(permutation_functions)
instance_urdf_paths = []
for i in tqdm(range(num_instances) , desc = "Generating URDFs"):
    instance_urdf_path = os.path.join(instances_folder, f"robot_{i}.urdf")
    modify_urdf_dynamics(robot_urdf_path, instance_urdf_path)
    instance_urdf_paths.append(instance_urdf_path)

# Convert each instance to USD
for i, instance_urdf_path in tqdm(enumerate(instance_urdf_paths) , desc = "Generating USDs"):
    dest_path = instance_urdf_path.replace(".urdf", ".usd")

    # Configure URDF to USD conversion
    urdf_converter_cfg = UrdfConverterCfg(
        asset_path=instance_urdf_path,
        usd_dir=os.path.dirname(dest_path),
        usd_file_name=os.path.basename(dest_path),
        force_usd_conversion=True,
        make_instanceable=False,
        import_inertia_tensor=True,
        fix_base=False,
        merge_fixed_joints=True,
        self_collision=False,
        default_drive_type="none",
        override_joint_dynamics=False,
        default_drive_stiffness=1.0,
        default_drive_damping=1.0,
        link_density=1.0,
        convex_decompose_mesh=False,
    )

    # Convert URDF to USD
    urdf_converter = UrdfConverter(urdf_converter_cfg)

    # Open the new stage (specified by urdf_converter.usd_path)
    stage_utils.open_stage(urdf_converter.usd_path)
    stage = omni.usd.get_context().get_stage()

    # Define a fixed joint for the base link
    default_prim = STAGE_PATH
    fixed_joint_path = f"{default_prim}/root_joint"
    define_prim(fixed_joint_path, "PhysicsFixedJoint")
    fixed_joint_prim = stage.GetPrimAtPath(fixed_joint_path)
    prim_utils.set_targets(fixed_joint_prim, "physics:body0", [f"{STAGE_PATH}/link_0"])

    # Get all joints under ArticulationRoot
    joint_prims = get_all_matching_child_prims(
        prim_path=default_prim, predicate=lambda a: "RevoluteJoint" in get_prim_type_name(a)
    )
    # Modify joint properties
    for joint_prim in joint_prims:
        joint_prim.GetAttribute("drive:angular:physics:maxForce").Set(100000.0)
        joint_prim.GetAttribute("drive:angular:physics:damping").Set(0.0)
        joint_prim.GetAttribute("drive:angular:physics:stiffness").Set(0.0)
        joint_prim.GetAttribute("physics:lowerLimit").Set(-np.inf)
        joint_prim.GetAttribute("physics:upperLimit").Set(np.inf)

    # Save the updated USD stage
    usd_save_path = urdf_converter.usd_path  # Save to the same path
    omni.usd.get_context().save_as_stage(usd_save_path)
    print(f"Saved updated USD stage to: {usd_save_path}")

# Trajectory generation
data = {
    "time": [],
    "mode": [],
    "Joint": [],
    "Position": [],
    "Velocity": [],
    "Acceleration": [],
    "Target_Torque_GC": [],
    "Effort_GC": [],
    "Target_Torque": [],
    "Effort": [],
}

dynamic_param_names = ["coloumbFriction" , "viscousFriction" , "ixx", "ixy", "ixz", "iyy", "iyz", "izz"]
dynamic_param_dict = {dynParam : [] for dynParam in dynamic_param_names}
data.update(dynamic_param_dict)

for perm_index, trajectory_function_list in tqdm(enumerate(permutation_functions) , desc = "Generating Permutations"):
    print([traj.__name__ for traj in trajectory_function_list])
    for instance_index in range(4):
        instance_urdf_path = instance_urdf_paths[4 * perm_index + instance_index]
        usd_path = instance_urdf_path.replace(".urdf", ".usd")

        # Open the new stage (specified by urdf_converter.usd_path)
        stage_utils.open_stage(usd_path)
        stage = omni.usd.get_context().get_stage()

        model = pin.buildModelFromUrdf(instance_urdf_path)
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

        mode = "".join(["1" if traj.__name__ == "sin_wave" else "0" for traj in trajectory_function_list])
        mode += f"{(instance_index):02b}"  # Add instance index as 2-bit binary

        last_joint_velocities = np.zeros(len(joint_prims))
        last_t = start_time

        # print(f"testing on {len(permutation_functions)} permutations")
        # print(f"currently on permutation : {perm_index}")
        # print([traj.__name__ for traj in trajectory_function_list])
        print(f"using instance: {4 * perm_index + instance_index}")

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

            coloumb_friction_coeff = []
            viscous_friction_coeff = []
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

                single_dynamic_parameters = instances_dynamic_params[4 * perm_index + instance_index][j]
                for idx , dynamic_param in enumerate(dynamic_param_names):
                    data[dynamic_param].append(single_dynamic_parameters[idx])

                coloumb_friction_coeff.append(single_dynamic_parameters[0])
                viscous_friction_coeff.append(single_dynamic_parameters[1])

            coloumb_friction_coeff = np.array(coloumb_friction_coeff)
            viscous_friction_coeff = np.array(viscous_friction_coeff)
            
            # Define the robot's configuration (joint positions) from Isaac Sim
            q = predict_position_midstep(np.array(joint_positions) , np.array(joint_velocities) , np.array(joint_accelerations) , dt)  * pi/180 # Use the joint angles read from Isaac Sim
            # Define the robot's velocity and acceleration (zero for gravity compensation)
            v = np.zeros(model.nv)  # Zero velocity
            a = np.zeros(model.nv)  # Zero acceleration

            # Compute gravity compensation torques using RNEA
            tau_gravity = pin.rnea(model, gravity_data, q, v, a)
            
            # Calculate the friction torque for each joint
            tau_coloumb_friction = -coloumb_friction_coeff * np.tanh(joint_velocities / 0.005)
            tau_viscous_friction = -viscous_friction_coeff * joint_velocities
            tau_friction = tau_coloumb_friction + tau_viscous_friction

            # Compute practical torques based on trajectory function
            tau_practical = [Amplitude_correcction[i] * traj_i(time.time() - start_time) for i , traj_i in enumerate(trajectory_function_list)]
            
            tau = tau_gravity + tau_practical + tau_viscous_friction + tau_friction

            # Apply the computed gravity compensation torques to the robot in Isaac Sim
            dc.set_articulation_dof_efforts(art, tau.tolist())

            if current_t - start_time >= time_per_perm:
                break

        simulation_context.stop()

# Save trajectory data
trajectory_folder = os.path.join(os.path.dirname(robot_urdf_path), "trajectories")
if not os.path.exists(trajectory_folder):
    os.makedirs(trajectory_folder)

# Save to CSV
df = pd.DataFrame(data)
df.to_csv(os.path.join(trajectory_folder , f"{STAGE_PATH.replace('/' , '')}_trajectory.csv" ), index=False)

del data

simulation_context.stop()
simulation_app.close()