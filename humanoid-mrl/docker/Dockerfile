ARG HOME="/root"
ARG BASE_STAGE="deploy"
# Compose mounts the colcon workspace at /workspace.
ARG ROS2_WS="/workspace"
ARG ROS2_WS_SRC="${ROS2_WS}/src"
ARG ROS2_WS_DEP="/root/workspace_deps"

# Simulation base

FROM ros:jazzy@sha256:be7350a0efd17925415a6a3a076d6009b425af7500b0a5cdb5d717e87ef46f39 AS sim
ARG HOME
ARG ROS2_WS
ARG ROS2_WS_SRC

SHELL ["/bin/bash", "-c"]

ARG UNITREE_SDK2_DIR="${HOME}/unitree_sdk2"

# Temporary until local examples replace Unitree SDK2 examples.
WORKDIR $UNITREE_SDK2_DIR
RUN git clone --depth 1 https://github.com/unitreerobotics/unitree_sdk2.git $UNITREE_SDK2_DIR
WORKDIR $UNITREE_SDK2_DIR/build
ENV LD_LIBRARY_PATH=/root/unitree_sdk2/thirdparty/lib/x86_64
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/unitree_robotics \
    && cmake --build . --target install -j"$(nproc)"

# Deployment base

FROM ros:jazzy@sha256:be7350a0efd17925415a6a3a076d6009b425af7500b0a5cdb5d717e87ef46f39 AS deploy
ARG HOME
ARG ROS2_WS
ARG ROS2_WS_SRC
ARG ROS2_WS_DEP

SHELL ["/bin/bash", "-c"]

RUN --mount=type=cache,id=core_apt_cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=core_apt_cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3-vcstool=0.3.* \
    ros-jazzy-pcl-ros=2.6.* \
    libpcl-dev=1.14.* \
    libomp-dev=1:18.0* \
    libeigen3-dev=3.4.* \
    libapr1-dev=1.7.* \
    libaprutil1-dev=1.6.* \
    ros-jazzy-librealsense2=2.58.* \
    ros-jazzy-realsense2-camera=4.58.* \
    ros-jazzy-realsense2-camera-msgs=4.58.* \
    ros-jazzy-realsense2-description=4.58.* \
    ros-jazzy-compressed-image-transport=4.0.* \
    && rm -rf /var/lib/apt/lists/*

# Dependency workspace
WORKDIR ${ROS2_WS_DEP}
COPY deploy_deps.repos deploy_deps.repos
RUN --mount=type=cache,id=core_colcon_deps_cache,target=${ROS2_WS_DEP}/build source /opt/ros/jazzy/setup.bash \
    && mkdir -p src \
    && vcs import --shallow --input deploy_deps.repos src \
    && colcon build

# Application

# hadolint ignore=DL3007
FROM ${BASE_STAGE} AS code
ARG HOME
ARG ROS2_WS
ARG ROS2_WS_SRC
ARG ROS2_WS_DEP

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV VENV_DIR=$HOME/venv \
    HOME=$HOME \
    ROS2_WS=$ROS2_WS \
    ROS2_WS_DEP=$ROS2_WS_DEP

RUN --mount=type=cache,id=core_apt_cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=core_apt_cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update && apt-get install -y --no-install-recommends \
    git=1:2.43.* \
    gdb=15.1* \
    build-essential=12.10* \
    cmake=3.28.* \
    ca-certificates \
    curl=8.5.* \
    libboost-all-dev=1.83.* \
    libxrandr-dev=2:1.5.* \
    libyaml-cpp-dev=0.8.* \
    libspdlog-dev=1:1.12.* \
    libglfw3-dev=3.3.* \
    python3-pip=24.0* \
    python3-tk=3.12.* \
    python3-venv=3.12.* \
    python3-opencv \
    ros-jazzy-rmw-cyclonedds-cpp=2.2.* \
    ros-jazzy-rosidl-generator-dds-idl=0.11.* \
    ros-jazzy-cv-bridge \
    ros-jazzy-topic-tools \
    ros-jazzy-pinocchio \
    ros-jazzy-rviz2 \
    ros-jazzy-rqt \
    ros-jazzy-rqt-common-plugins \
    ros-jazzy-usb-cam \
    ros-jazzy-mocap4r2-msgs \
    v4l-utils \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

COPY .vscode $HOME/.vscode

# Python environment
WORKDIR ${HOME}
COPY requirements.txt docker/env.bash ${HOME}/
# hadolint ignore=DL3042,SC2016
RUN --mount=type=cache,id=core_pip_cache,target=/root/.cache/pip \
    python3 -m venv --symlinks --system-site-packages ${VENV_DIR} \
    && source ${VENV_DIR}/bin/activate \
    && pip install -r requirements.txt \
    && rm requirements.txt \
    && echo 'source ${HOME}/env.bash' >>${HOME}/.bashrc

# GR00T client
RUN --mount=type=cache,id=core_pip_cache,target=/root/.cache/pip \
    source ${VENV_DIR}/bin/activate \
    && pip install pyzmq==27.1.0 numpy==1.26.4 msgpack==1.1.2 scipy==1.11.4 \
    && git clone --depth 1 https://github.com/NVIDIA/Isaac-GR00T.git /tmp/Isaac-GR00T \
    && sed -i 's/==3.10.\*/>=3.10/g' /tmp/Isaac-GR00T/pyproject.toml /tmp/Isaac-GR00T/setup.py 2>/dev/null || true \
    && python -c "import pathlib; p = pathlib.Path('/tmp/Isaac-GR00T/gr00t/policy/__init__.py'); lines = [l for l in p.read_text().splitlines() if 'Gr00tPolicy' not in l]; func = 'def __getattr__(name):\n    if name == \"Gr00tPolicy\":\n        from .gr00t_policy import Gr00tPolicy\n        return Gr00tPolicy\n    raise AttributeError(f\"module {__name__!r} has no attribute {name!r}\")'; p.write_text('\\n'.join(lines) + '\\n' + func)" \
    && pip install /tmp/Isaac-GR00T --no-deps \
    && rm -rf /tmp/Isaac-GR00T

# Build colcon workspace
# Source is bind-mounted; dependencies are prebuilt.

WORKDIR ${ROS2_WS}
# Two workers avoid OOM failures on the Jetson.
RUN --mount=type=cache,id=core_colcon_cache,target=${ROS2_WS}/build source ${HOME}/env.bash \
    && colcon build --parallel-workers 2 --packages-skip $(colcon list --base-paths ${ROS2_WS_SRC}/nvblox_ros2/ | awk '{print $1}')

# Documentation

FROM code AS documentation

ARG HOME

SHELL ["/bin/bash", "-c"]

WORKDIR ${HOME}
COPY docs/requirements.txt ${HOME}/
RUN source env.bash && pip install --no-cache-dir -r requirements.txt && rm requirements.txt
