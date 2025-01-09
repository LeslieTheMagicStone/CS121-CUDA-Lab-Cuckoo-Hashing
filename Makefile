CC=g++
NVCC=nvcc
CXXFLAGS=-std=c++11 -O3

all: experiments

experiments: experiments.cu parallel.cuh vanilla.hpp
	${NVCC} $< -o $@ ${CXXFLAGS}
	./experiments
	clean

.PHONY: clean
clean:
	rm -f experiments