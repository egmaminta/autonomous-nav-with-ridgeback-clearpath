"""Run the full ridgeback_autonav stack against the built-in 2D simulator (no hardware).

  # torch-free: ground-truth fake perception drives the whole mission FSM
  ros2 launch ridgeback_autonav_sim sim.launch.py task:="Go to Room 206"

  # exercise the REAL perception_node (YOLO+PARSeq) on the synthetic camera
  ros2 launch ridgeback_autonav_sim sim.launch.py use_real_perception:=true

Brings up: sim_node + topic_relay + mapping + nav_server + explorer + perception
(fake or real) + mission. TF for the broadcasters/listeners is remapped onto the
robot's namespaced /r100_0140/tf so the whole tree is consistent (the fix that
also matters on the real robot).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    bringup = get_package_share_directory('ridgeback_autonav_bringup')
    sim_share = get_package_share_directory('ridgeback_autonav_sim')
    params = os.path.join(bringup, 'config', 'ridgeback.yaml')
    world = os.path.join(sim_share, 'config', 'sim_world.yaml')

    params_file = LaunchConfiguration('params_file')
    world_file = LaunchConfiguration('world_file')
    task = LaunchConfiguration('task')
    tf = LaunchConfiguration('tf_topic')
    tf_static = LaunchConfiguration('tf_static_topic')

    args = [
        DeclareLaunchArgument('params_file', default_value=params),
        DeclareLaunchArgument('world_file', default_value=world),
        DeclareLaunchArgument('task', default_value='Go to Room 206'),
        DeclareLaunchArgument('use_real_perception', default_value='false'),
        DeclareLaunchArgument('camera_enabled', default_value='true'),
        DeclareLaunchArgument('tf_topic', default_value='/r100_0140/tf'),
        DeclareLaunchArgument('tf_static_topic', default_value='/r100_0140/tf_static'),
    ]
    tf_remaps = [('/tf', tf), ('/tf_static', tf_static)]

    def cond(name, positive=True):
        expr = PythonExpression(["'", LaunchConfiguration(name), "' == 'true'"])
        return IfCondition(expr) if positive else UnlessCondition(expr)

    nodes = [
        Node(package='ridgeback_autonav_sim', executable='sim_node', name='sim_node',
             parameters=[params, {'world_file': world_file,
                                  'camera_enabled': LaunchConfiguration('camera_enabled')}],
             remappings=tf_remaps, output='screen'),
        Node(package='ridgeback_autonav_bringup', executable='topic_relay', name='topic_relay',
             parameters=[params], output='screen'),
        Node(package='ridgeback_autonav_nav', executable='mapping_node', name='mapping_node',
             parameters=[params], remappings=tf_remaps, output='screen'),
        Node(package='ridgeback_autonav_nav', executable='nav_server_node', name='nav_server_node',
             parameters=[params], output='screen'),
        Node(package='ridgeback_autonav_nav', executable='explorer_node', name='explorer_node',
             parameters=[params], output='screen'),
        # Fake (ground-truth) perception — default, torch-free.
        Node(package='ridgeback_autonav_sim', executable='fake_perception_node', name='fake_perception_node',
             parameters=[params, {'world_file': world_file}], output='screen',
             condition=cond('use_real_perception', positive=False)),
        # Real perception on the synthetic camera (needs torch + YOLO/PARSeq).
        Node(package='ridgeback_autonav_perception', executable='perception_node', name='perception_node',
             parameters=[params], remappings=tf_remaps, output='screen',
             condition=cond('use_real_perception')),
        Node(package='ridgeback_autonav_mission', executable='mission_node', name='mission_node',
             parameters=[params, {'task': task}], output='screen'),
    ]
    return LaunchDescription(args + nodes)
