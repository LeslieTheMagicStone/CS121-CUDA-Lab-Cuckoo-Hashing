CC=g++
NVCC=nvcc
CXXFLAGS=-std=c++11 -O3

all: experiments

experiments: build-experiments
			./build-experiments

build-experiments:experiments.cu parallel.cuh vanilla.hpp
	${NVCC} $< -o $@ ${CXXFLAGS}

.PHONY: clean
clean:
	rm -f build-experiments