"""Bring up the full ridgeback_autonav autonomy stack on the Ridgeback.

Each subsystem can be toggled; all share one params file. Typical use:

  ros2 launch ridgeback_autonav_bringup bringup.launch.py task:="Go to Room 206"

For perception-only tuning against a bag, disable the rest:

  ros2 launch ridgeback_autonav_bringup bringup.launch.py \
      enable_relay:=false enable_mapping:=false enable_nav:=false \
      enable_explorer:=false enable_mission:=false
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('ridgeback_autonav_bringup')
    default_params = os.path.join(pkg, 'config', 'ridgeback.yaml')

    params_file = LaunchConfiguration('params_file')
    task = LaunchConfiguration('task')
    tf = LaunchConfiguration('tf_topic')
    tf_static = LaunchConfiguration('tf_static_topic')

    args = [
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('task', default_value='Go to Room 206'),
        # The Ridgeback publishes TF on a namespaced topic; our map->odom
        # broadcaster (mapping) and the perception TF listener must use the same
        # one or the tree splits and map<-camera lookups fail. Set to '/tf' for a
        # robot that uses the default tf topics.
        DeclareLaunchArgument('tf_topic', default_value='/r100_0140/tf'),
        DeclareLaunchArgument('tf_static_topic', default_value='/r100_0140/tf_static'),
        DeclareLaunchArgument('enable_relay', default_value='true'),
        DeclareLaunchArgument('enable_mapping', default_value='true'),
        DeclareLaunchArgument('enable_nav', default_value='true'),
        DeclareLaunchArgument('enable_explorer', default_value='true'),
        DeclareLaunchArgument('enable_perception', default_value='true'),
        DeclareLaunchArgument('enable_mission', default_value='true'),
    ]
    tf_remaps = [('/tf', tf), ('/tf_static', tf_static)]

    def cond(name):
        return IfCondition(PythonExpression(["'", LaunchConfiguration(name), "' == 'true'"]))

    nodes = [
        Node(
            package='ridgeback_autonav_bringup', executable='topic_relay', name='topic_relay',
            parameters=[params_file], output='screen', condition=cond('enable_relay'),
        ),
        Node(
            package='ridgeback_autonav_nav', executable='mapping_node', name='mapping_node',
            parameters=[params_file], remappings=tf_remaps, output='screen',
            condition=cond('enable_mapping'),
        ),
        Node(
            package='ridgeback_autonav_nav', executable='nav_server_node', name='nav_server_node',
            parameters=[params_file], output='screen', condition=cond('enable_nav'),
        ),
        Node(
            package='ridgeback_autonav_nav', executable='explorer_node', name='explorer_node',
            parameters=[params_file], output='screen', condition=cond('enable_explorer'),
        ),
        Node(
            package='ridgeback_autonav_perception', executable='perception_node', name='perception_node',
            parameters=[params_file], remappings=tf_remaps, output='screen',
            condition=cond('enable_perception'),
        ),
        Node(
            package='ridgeback_autonav_mission', executable='mission_node', name='mission_node',
            parameters=[params_file, {'task': task}], output='screen',
            condition=cond('enable_mission'),
        ),
    ]

    return LaunchDescription(args + nodes)
