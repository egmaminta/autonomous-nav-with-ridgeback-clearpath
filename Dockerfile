# ridgeback_autonav simulator image: ROS 2 Humble + the ridgeback_autonav workspace, built and ready to
# run the self-contained 2D simulator (no hardware, no Gazebo, no Nav2).
#
#   Default build  : torch-free. The mission FSM is driven by fake_perception
#                    (ground-truth signs). This is all you need to see the
#                    EXPLORE -> APPROACH -> CONFIRM -> RETURN_HOME -> DONE loop.
#
#   Optional build : add the real YOLO+PARSeq perception_node on the synthetic
#                    camera with  --build-arg INSTALL_PERCEPTION=true  (~2 GB:
#                    pulls torch/torchvision). Then launch with
#                    use_real_perception:=true.
#
# Build (default):   docker build -t ridgeback-autonav-sim:humble .
# Build (real perc): docker build --build-arg INSTALL_PERCEPTION=true -t ridgeback-autonav-sim:perception .
# Run:               docker run --rm -it ridgeback-autonav-sim:humble
#
# ros-base already provides: rclpy, ros2 launch, common_interfaces
# (std/geometry/sensor/nav/visualization_msgs), tf2_ros/tf2_geometry_msgs,
# message_filters, and the rosidl message-generation toolchain.
FROM ros:humble-ros-base AS base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# colcon (build tool, not in the base image) + cv_bridge (the sim's synthetic
# RGB-D camera) + foxglove_bridge (so you can *watch* the sim from Foxglove
# Studio over ws://localhost:8765 — no X11 needed on macOS). Everything else the
# default sim needs ships in ros-base.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-colcon-common-extensions \
        ros-humble-cv-bridge \
        ros-humble-foxglove-bridge \
    && rm -rf /var/lib/apt/lists/*

# Optional real-perception extras. ultralytics pulls in torch + torchvision.
# Skipped unless you pass --build-arg INSTALL_PERCEPTION=true.
ARG INSTALL_PERCEPTION=false
RUN if [ "$INSTALL_PERCEPTION" = "true" ]; then \
        pip3 install --no-cache-dir ultralytics huggingface_hub pillow dill ; \
    fi

# Optional: RViz2 for the X11/XQuartz workflow (adds Qt — a few hundred MB).
# Foxglove (above) is the recommended viewer on macOS; this is only for people
# who specifically want native RViz over XQuartz.
ARG INSTALL_RVIZ=false
RUN if [ "$INSTALL_RVIZ" = "true" ]; then \
        apt-get update && apt-get install -y --no-install-recommends ros-humble-rviz2 \
        && rm -rf /var/lib/apt/lists/* ; \
    fi

# Optional: rosboard — a browser-based dashboard (http://localhost:8888, no X11,
# no desktop app). Cloned + built as a separate colcon overlay at /opt/rosboard_ws
# so it never touches the main workspace.
ARG INSTALL_ROSBOARD=false
RUN if [ "$INSTALL_ROSBOARD" = "true" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
            git python3-tornado python3-simplejson python3-pil \
        && rm -rf /var/lib/apt/lists/* \
        && git clone --depth 1 https://github.com/dheera/rosboard.git /opt/rosboard_ws/src/rosboard \
        && source /opt/ros/humble/setup.bash \
        && ( cd /opt/rosboard_ws && colcon build ) ; \
    fi

# Optional: RL / world-model training deps. Installs Gymnasium so the ROS-free
# environment in ridgeback_autonav_sim/rl_env.py (RidgebackAutoNav-v1) is importable. Heavier RL
# libraries (stable-baselines3 + torch) are intentionally NOT installed here —
# add them yourself when you want to train: pip3 install "stable-baselines3[extra]".
ARG INSTALL_RL=false
RUN if [ "$INSTALL_RL" = "true" ]; then \
        apt-get update && apt-get install -y --no-install-recommends python3-pip \
        && rm -rf /var/lib/apt/lists/* \
        && pip3 install --no-cache-dir "gymnasium>=0.29" ; \
    fi

# --- build the workspace ---------------------------------------------------
WORKDIR /ws
# Copy only the source tree (see .dockerignore — .venv, .git, build artifacts
# are excluded so the build context stays small).
COPY src ./src
# colcon resolves build order itself (ridgeback_autonav_msgs is generated first, then the
# pure-python packages that depend on it).
RUN source /opt/ros/humble/setup.bash \
    && colcon build --executor sequential

# Source ROS 2 + the workspace overlay for every container command.
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Default: full mission in sim, torch-free. Override the task (or anything else)
# by appending your own command, e.g.
#   docker run --rm -it ridgeback-autonav-sim:humble \
#     ros2 launch ridgeback_autonav_sim sim.launch.py task:="Go to Room 101"
CMD ["ros2", "launch", "ridgeback_autonav_sim", "sim.launch.py", "task:=Go to Room 206"]
