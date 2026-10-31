# Snake AI Makefile
# Author: Miguel Mochizuki Silva
# Description: Thin wrapper around CMake: build, test and clean

BUILD ?= build

.PHONY: all test clean

all:
	cmake -S . -B $(BUILD) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD) -j

test: all
	ctest --test-dir $(BUILD) --output-on-failure

clean:
	rm -rf $(BUILD)
