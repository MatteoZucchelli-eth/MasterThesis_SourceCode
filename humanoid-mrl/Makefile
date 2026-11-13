.DEFAULT_GOAL := help

COMPOSE := docker compose -f docker/docker-compose.yaml
SIM_TYPE ?= sim-isaacsim
DEBUG_ROUTE_NET ?= 192.168.123.0/24
DEBUG_ROUTE_VIA ?= 10.160.72.194

# Simulation argument shorthands; command-line values must be preserved.
ifdef WALK
override SIM_EXTRA_ARGS := --cfg.walk-mode $(SIM_EXTRA_ARGS)
endif
ifdef DRILL_CONTACT
override SIM_EXTRA_ARGS := --cfg.no-disable-drill-bit-collision $(SIM_EXTRA_ARGS)
endif
ifdef CALIBRATION
override SIM_EXTRA_ARGS := --cfg.calibration $(SIM_EXTRA_ARGS)
endif
ifdef CALIBRATION_RPY
override SIM_EXTRA_ARGS := --cfg.calibration-rpy-deg $(CALIBRATION_RPY) $(SIM_EXTRA_ARGS)
endif
ifdef CALIBRATION_XYZ
override SIM_EXTRA_ARGS := --cfg.calibration-xyz-m $(CALIBRATION_XYZ) $(SIM_EXTRA_ARGS)
endif
export SIM_EXTRA_ARGS

# Bind the interface routing to DEBUG_PEER and use unicast DDS discovery.
ifdef DEBUG_PEER
DEBUG_IFACE := $(shell ip -o route get $(DEBUG_PEER) 2>/dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p')
ifeq ($(DEBUG_IFACE),)
$(error DEBUG_PEER=$(DEBUG_PEER) is not routable)
endif
CYCLONEDDS_URI := <CycloneDDS><Domain id="any"><General><Interfaces><NetworkInterface name="lo" priority="default" multicast="default" /><NetworkInterface name="$(DEBUG_IFACE)" priority="10" multicast="default" /></Interfaces></General><Discovery><Peers><Peer address="$(DEBUG_PEER)"/></Peers></Discovery></Domain></CycloneDDS>
export CYCLONEDDS_URI
$(info DDS: lo + $(DEBUG_IFACE), peer $(DEBUG_PEER))
endif

TARGETS := help control-sim stop-control-sim teleoperation stop-teleoperation \
	control stop-control teleop-real \
	stop-teleop-real deployment stop-deployment stop environment \
	stop-environment vla-logs simulation sync-cameras build-docs \
	live-debug stop-live-debug
.PHONY: $(TARGETS)

help: ## List targets and options
	@awk 'BEGIN {FS = ":.*## "; printf "Usage: make <target> [OPTION=value]\n\n"} \
		/^[a-zA-Z0-9_-]+:.*## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf '\nOptions: SIM_TYPE, SIM_EXTRA_ARGS, WALK, DRILL_CONTACT, CALIBRATION, CALIBRATION_RPY, CALIBRATION_XYZ, DEBUG_PEER\n'

control-sim: ## Start the development container and Isaac Sim
	$(COMPOSE) --profile control-sim up --build -d

stop-control-sim: ## Stop the development container and Isaac Sim
	$(COMPOSE) --profile control-sim down

teleoperation: ## Start the full simulated teleoperation stack
	$(COMPOSE) --profile teleoperation up --build -d

stop-teleoperation: ## Stop the simulated teleoperation stack
	$(COMPOSE) --profile teleoperation down

control: ## Start only the development/control container
	$(COMPOSE) --profile control up --build -d

stop-control: ## Stop the development/control container
	$(COMPOSE) --profile control down

teleop-real: ## Start teleoperation on the G1
	$(COMPOSE) --profile teleop-real up --build -d

stop-teleop-real: ## Stop real-robot teleoperation
	$(COMPOSE) --profile teleop-real down

deployment: ## Start the G1 deployment stack
	$(COMPOSE) --profile deployment up --build -d
	$(COMPOSE) --profile deployment up -d --force-recreate --no-deps \
		realsense-node g1-hand-cam-deploy

stop-deployment: ## Stop the G1 deployment stack
	$(COMPOSE) --profile deployment down

stop: ## Stop development and simulation containers
	$(COMPOSE) --profile "*" down

environment: ## Start the simulation environment
	$(COMPOSE) --profile environment up --build -d

stop-environment: ## Stop the simulation environment
	$(COMPOSE) --profile environment down

vla-logs: ## Follow VLA server logs
	$(COMPOSE) --profile vla logs -f vla-server

simulation: ## Run the selected simulator (SIM_TYPE=sim-isaacsim|sim-mujoco)
	$(COMPOSE) --profile sim --profile $(SIM_TYPE) up --build

sync-cameras: ## Sync URDF camera poses into the Isaac asset (ARGS=--check)
	$(COMPOSE) run --rm --no-deps \
		--entrypoint /workspace/scripts/isaac_usd_python.sh simulation-isaacsim \
		/workspace/src/g1_simulation/scripts/sync_camera_extrinsics_to_usd.py $(ARGS)

build-docs: ## Build and open the documentation
	$(COMPOSE) up build-docs --build
	open docs/html/index.html

live-debug: ## Start the CPU-only ROS debugging container
	xhost +local:docker
	@if [ -z "$(DEBUG_ROUTE_NET)" ]; then \
		:; \
	elif ip route show "$(DEBUG_ROUTE_NET)" | grep -q .; then \
		echo "Route $(DEBUG_ROUTE_NET) already present"; \
	else \
		sudo ip route add "$(DEBUG_ROUTE_NET)" via "$(DEBUG_ROUTE_VIA)" \
			|| echo "Warning: could not add route $(DEBUG_ROUTE_NET) via $(DEBUG_ROUTE_VIA)"; \
	fi
	$(COMPOSE) --profile live-debug up --build -d
	@docker exec humanoid-mrl-live-debug pgrep -x rviz2 >/dev/null 2>&1 \
		|| docker exec -d humanoid-mrl-live-debug \
			bash -lc 'rviz2 -d /workspace/config.rviz'
	@echo 'live-debug is ready; attach with: docker exec -it humanoid-mrl-live-debug bash'

stop-live-debug: ## Stop the ROS debugging container
	$(COMPOSE) --profile live-debug down
