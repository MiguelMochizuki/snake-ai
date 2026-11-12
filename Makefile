# Snake AI Makefile
# Author: Miguel Mochizuki Silva
# Description: Wrapper around CMake: build (with or without the SDL window), test, watch a network play, make the GIF

# ==============================
# Configuration
# ==============================

# Two build variants, each in its own directory so they never clash:
#   terminal  build/      everything, no extra dependencies (BUILD=dir overrides)
#   gui       build-sdl/  the same plus watch --gui and --frames (fetches SDL3 and miniaudio)
BUILD ?= build
GUI_BUILD = build-sdl
CMAKE = cmake
CMAKE_FLAGS = -DCMAKE_BUILD_TYPE=Release

# Settings for watch and gif (override on the command line, e.g. make watch SEED=3)
CHECKPOINT ?= models/es_finetuned.ckpt
# KIND: es for es_finetuned.ckpt, astar for imitation_astar.ckpt
KIND ?= es
SEED ?= 1
MS ?= 50
FRAMES_DIR ?= frames
GIF ?= docs/astar_network.gif

# Files that trigger a rebuild; deps/ and build trees are not scanned
SOURCES = $(shell find src include apps experiments tests -name '*.cpp' -o -name '*.hpp' 2>/dev/null)

# Colors
GREEN = \033[0;32m
BLUE = \033[0;34m
YELLOW = \033[0;33m
NC = \033[0m

# Configures and builds one variant. Arguments: label, build directory, extra CMake flags
define build_variant
	@echo "$(BLUE)Configuring CMake ($(1))...$(NC)"
	@$(CMAKE) -S . -B $(2) $(CMAKE_FLAGS) $(3)
	@echo "$(BLUE)Building project...$(NC)"
	@$(CMAKE) --build $(2) -j
	@touch $(2)/.built
	@echo "$(GREEN)Build complete: $(2)/bin$(NC)"
endef

# ==============================
# Dependencies
# ==============================

# The submodules are fetched on first use
deps/snake-game/CMakeLists.txt deps/nn-cpp/CMakeLists.txt:
	@echo "$(BLUE)Fetching submodules...$(NC)"
	@git submodule update --init --recursive

# ==============================
# Build targets
# ==============================

.PHONY: all gui
all: $(BUILD)/.built
gui: $(GUI_BUILD)/.built

# Each variant rebuilds only if sources or CMake files changed. The .built stamp is written only
# after the whole build succeeded, so a failed build is never mistaken for a finished one
$(BUILD)/.built: $(SOURCES) CMakeLists.txt apps/CMakeLists.txt tests/CMakeLists.txt deps/snake-game/CMakeLists.txt deps/nn-cpp/CMakeLists.txt
	$(call build_variant,terminal,$(BUILD),)

$(GUI_BUILD)/.built: $(SOURCES) CMakeLists.txt apps/CMakeLists.txt tests/CMakeLists.txt deps/snake-game/CMakeLists.txt deps/nn-cpp/CMakeLists.txt
	$(call build_variant,with GUI,$(GUI_BUILD),-DSNAKE_AI_BUILD_GUI=ON)

# ==============================
# Test targets
# ==============================

.PHONY: test test-gui test-all
test: all
	@echo "$(BLUE)Running tests (terminal)...$(NC)"
	@ctest --test-dir $(BUILD) --output-on-failure

test-gui: gui
	@echo "$(BLUE)Running tests (with GUI)...$(NC)"
	@ctest --test-dir $(GUI_BUILD) --output-on-failure

test-all: test test-gui

# ==============================
# Watch a network play
# ==============================

.PHONY: watch watch-gui gif
watch: all
	@$(BUILD)/bin/watch $(CHECKPOINT) $(SEED) $(MS) 1 $(KIND)

watch-gui: gui
	@$(GUI_BUILD)/bin/watch $(CHECKPOINT) $(SEED) $(MS) 1 $(KIND) --gui

# Remakes the README GIF from off-screen frames (needs ffmpeg)
gif: gui
	@echo "$(BLUE)Drawing frames...$(NC)"
	@rm -rf $(FRAMES_DIR) && mkdir -p $(FRAMES_DIR)
	@$(GUI_BUILD)/bin/watch $(CHECKPOINT) $(SEED) $(MS) 125 $(KIND) --frames $(FRAMES_DIR)
	@ffmpeg -loglevel error -y -framerate 25 -i $(FRAMES_DIR)/frame-%05d.ppm \
		-vf "crop=640:662:0:0,scale=360:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=24:stats_mode=diff[p];[b][p]paletteuse=dither=none" \
		-loop 0 $(GIF)
	@rm -rf $(FRAMES_DIR)
	@echo "$(GREEN)Wrote $(GIF)$(NC)"

# ==============================
# Clean
# ==============================

.PHONY: clean rebuild
clean:
	@echo "$(YELLOW)Cleaning build files...$(NC)"
	@rm -rf $(BUILD) $(GUI_BUILD) $(FRAMES_DIR)
	@echo "$(YELLOW)Clean complete!$(NC)"

rebuild: clean all

# ==============================
# Help
# ==============================

.PHONY: help
help:
	@echo "$(GREEN)Snake AI - Makefile Options$(NC)"
	@echo ""
	@echo "$(BLUE)Build (only if changed; submodules are fetched automatically):$(NC)"
	@echo "  make                - Everything, terminal only: $(BUILD)/bin"
	@echo "  make gui            - Everything plus the SDL window: $(GUI_BUILD)/bin"
	@echo ""
	@echo "$(BLUE)Test:$(NC)"
	@echo "  make test           - Build and run the tests"
	@echo "  make test-gui       - Same, for the GUI build"
	@echo "  make test-all       - Both"
	@echo ""
	@echo "$(BLUE)Watch a network play (CHECKPOINT, KIND, SEED and MS can be overridden; the default is the ES network):$(NC)"
	@echo "  make watch          - In the terminal"
	@echo "  make watch-gui      - In a window"
	@echo "  make gif            - Remake $(GIF) with ffmpeg"
	@echo ""
	@echo "$(BLUE)Clean:$(NC)"
	@echo "  make clean          - Remove the build directories"
	@echo "  make rebuild        - Clean and rebuild"
