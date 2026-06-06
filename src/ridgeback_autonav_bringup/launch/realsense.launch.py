"""Run the Intel RealSense D435 driver ON THE JETSON (where the camera + our
perception_node live), publishing under the robot's camera_0 namespace.

Why this exists: the D435 is plugged into the Jetson, but the platform's stock
camera driver runs on the robot's other PC and can't see the device. We run the
driver locally instead.

Key choices:
  * align_depth.enable=true  -> depth is registered to the COLOR image, so
    perception can sample depth at a color-pixel bbox (our projection assumes this).
  * pointcloud.enable=false  -> saves USB/CPU bandwidth (we don't use it).
  * initial_reset=true       -> hardware-resets the device on start, clearing the
    "driver up but no streams" stuck state.

Requires the driver (one-time):
  sudo apt install ros-humble-realsense2-camera
Plug the D435 into a Jetson USB-3 (SuperSpeed/blue) port.

  ros2 launch ridgeback_autonav_bringup realsense.launch.py
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    rs_launch = PathJoinSubstitution(
        [FindPackageShare('realsense2_camera'), 'launch', 'rs_launch.py'])
    return LaunchDescription([
        DeclareLaunchArgument('camera_namespace', default_value='r100_0140/sensors'),
        DeclareLaunchArgument('camera_name', default_value='camera_0'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(rs_launch),
            launch_arguments={
                'camera_namespace': LaunchConfiguration('camera_namespace'),
                'camera_name': LaunchConfiguration('camera_name'),
                'enable_color': 'true',
                'enable_depth': 'true',
                'align_depth.enable': 'true',
                'pointcloud.enable': 'false',
                'initial_reset': 'true',
            }.items()),
    ])
