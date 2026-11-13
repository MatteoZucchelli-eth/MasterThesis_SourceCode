# GR00T eval-server launcher. Wraps docker compose so the venv volume is reused
# instead of leaked. Pass MODEL_PATH (and optionally EMBODIMENT_TAG) as arguments.
#
#   make up      MODEL_PATH=general_3k              # start server (foreground, Ctrl-C stops)
#   make up-d    MODEL_PATH=checkpoints-general_3k  # start VLA server (detached)
#   make up-d    MODEL_PATH=perceptionv1            # auto-detects -> perception server
#   make up-d    MODEL_PATH=x SERVER=perception     # force the perception server
#   make down                                       # stop + remove container (keeps venv)
#
# TWO MODELS AT ONCE: each PORT gets its own compose project (PROJECT=gr00t-$PORT),
# so the two containers coexist and share the one venv volume. Pass the same PORT to
# every follow-up target to address that server:
#
#   make up-d MODEL_PATH=checkpoints-red-small-predict-1500 PORT=5555
#   make up-d MODEL_PATH=perceptionv1                       PORT=5556
#   make logs PORT=5556        # follow just the perception server
#   make down PORT=5556        # stop just the perception server
#   make shell                                      # interactive bash in the container
#   make build                                      # (re)build image from docker/Dockerfile
#   make logs                                       # follow logs
#   make clean                                      # remove container + venv volume + prune leaks
#   make prune                                      # reclaim historical dangling 15G volumes
#
# Which server comes up (scripts/serve_entrypoint.sh):
#   SERVER=auto (default) inspects <MODEL_PATH>/config.json — a Gr00tN1d7Perception
#   checkpoint launches run_perception_server.py, anything else run_gr00t_server.py.
#   Override with SERVER=perception or SERVER=vla.

MODEL_PATH     ?= general_3k
EMBODIMENT_TAG ?= NEW_EMBODIMENT
SERVER         ?= auto
PORT           ?= 5555
# One compose project per PORT, so two models can be served side by side without
# either `up` replacing the other's container. Every target keys off PORT, so
# `make logs PORT=5556` / `make down PORT=5556` address the right one. The venv and
# vscode-server volumes are pinned by name in docker-compose.yml and stay shared.
PROJECT        ?= gr00t-$(PORT)
# Newest installed Claude Code native binary from the VS Code extension (survives updates).
CLAUDE_BIN ?= $(shell ls -1t $(HOME)/.vscode-server/extensions/anthropic.claude-code-*-linux-x64/resources/native-binary/claude 2>/dev/null | head -1)
COMPOSE := MODEL_PATH=$(MODEL_PATH) EMBODIMENT_TAG=$(EMBODIMENT_TAG) SERVER=$(SERVER) PORT=$(PORT) CLAUDE_BIN=$(CLAUDE_BIN) docker compose -p $(PROJECT)

.DEFAULT_GOAL := help
.PHONY: help up up-d down attach claude shell build logs ps seed-vscode clean prune

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "  Current: MODEL_PATH=$(MODEL_PATH)  EMBODIMENT_TAG=$(EMBODIMENT_TAG)  PORT=$(PORT)  PROJECT=$(PROJECT)"

up: ## Start the eval server in the foreground (Ctrl-C stops it cleanly)
	$(COMPOSE) up --remove-orphans

up-d: ## Start the eval server detached (then `make attach` or attach VS Code)
	$(COMPOSE) up -d --remove-orphans

down: ## Stop and remove the container (keeps venv + vscode-server volumes)
	$(COMPOSE) down --remove-orphans

attach: ## Attach an interactive shell to the RUNNING container (fast; claude on PATH)
	$(COMPOSE) exec gr00t bash

claude: ## Launch Claude Code inside the running container
	$(COMPOSE) exec gr00t claude

shell: ## Open a shell in a throwaway container (use `attach` for the running one)
	$(COMPOSE) run --rm gr00t bash

build: ## Build the gr00t image from docker/Dockerfile (honours .dockerignore)
	$(COMPOSE) build

logs: ## Follow the running container's logs
	$(COMPOSE) logs -f

ps: ## Show container status
	$(COMPOSE) ps

seed-vscode: ## Pre-load the vscode-server volume from the host so even the FIRST attach is instant
	$(COMPOSE) run --rm -v $(HOME)/.vscode-server:/host-vscode:ro gr00t \
		bash -c 'cp -an /host-vscode/. /root/.vscode-server/ && echo "vscode-server volume seeded"'

clean: ## Full teardown: remove container + venv AND vscode-server volumes, then prune dangling
	$(COMPOSE) down -v --remove-orphans
	docker volume prune -f

prune: ## Reclaim historical dangling anonymous volumes (the leftover ~15G ones)
	docker volume prune -f
