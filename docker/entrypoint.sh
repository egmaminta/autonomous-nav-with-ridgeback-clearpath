#!/usr/bin/env bash
# Source ROS 2 Humble + the ridgeback_autonav workspace overlay, then exec the given command.
# Used as the image ENTRYPOINT so `docker run <img> ros2 ...` / `docker compose
# exec sim ros2 ...` both come up with the environment already sourced.
set -e
source /opt/ros/humble/setup.bash
if [ -f /ws/install/setup.bash ]; then
  source /ws/install/setup.bash
fi
exec "$@"
