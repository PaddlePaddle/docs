# Heap Memory Profiling and Optimization

Any computer program has the danger of memory leak. Generally, **Memory Leak** is caused by the unreleased heap memory allocated by the program. As the memory occupied by the program becomes larger and larger, it will affect the stability of the program, which may make the running speed slower or give rise to OoM(Out of Memory). It even compromises the stability of the machine in use, and leads to *downtime* .


There are many memory leak analysis tools at present. Typical ones include, [valgrind](http://valgrind.org/docs/manual/quick-start.html#quick-start.intro), [gperftools](https://gperftools.github.io/gperftools/).

Because Fluid runs in C++ core driven by Python, It is very difficult for valgrind to analyze directly. You need to compile the debug version and dedicated Python version with valgrind support, and most of the output information is Python's own symbols and call information. In addition, valgrind will make the program run very slowly, so it is not recommended.

Here we mainly introduce the use of [gperftools](https://gperftools.github.io/gperftools/) .

gperftool mainly supports four functions:

- thread-caching malloc
- heap-checking using tcmalloc
- heap-profiling using tcmalloc
- CPU profiler

Paddle also provides a [tutorial on CPU performance analysis](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/howto/optimization/cpu_profiling_en.md) based on gperftool.

For the analysis for heap, we mainly use thread-caching malloc and heap-profiling using tcmalloc.

## Environment

This tutorial is based on the Docker development environment paddlepaddle/paddle:latest-dev provided by paddle, based on the Ubuntu 16.04.4 LTS environment.

## Manual

- Install google-perftools

```
apt-get install libunwind-dev 
apt-get install google-perftools
```

- Install pprof

```
go get -u github.com/google/pprof
```

- Configure Running Environment

```
export PPROF_PATH=/root/gopath/bin/pprof
export PPROF_BINARY_PATH=/root/gopath/bin/pprof
export LD_PRELOAD=/usr/lib/libtcmalloc.so.4
```

- Use heap profile to run python program. The essence of it is to get a snapshot of the heap allocation periodically.

```
# HEAPPROFILE sets the directory and file prefix of the generated heap analysis file
# HEAP_PROFILE_ALLOCATION_INTERVAL Sets how many storage dumps are allocated for each dump, default 1GB
env HEAPPROFILE="./perf_log/test.log" HEAP_PROFILE_ALLOCATION_INTERVAL=209715200 python trainer.py
```

As the program runs, a lot of files will be generated in the perf_log folder as follows:

```
-rw-r--r-- 1 root root 1.0M Jun  1 15:00 test.log.0001.heap
-rw-r--r-- 1 root root 1.0M Jun  1 15:00 test.log.0002.heap
-rw-r--r-- 1 root root 1.0M Jun  1 15:00 test.log.0003.heap
-rw-r--r-- 1 root root 1.0M Jun  1 15:00 test.log.0004.heap
-rw-r--r-- 1 root root 1.0M Jun  1 15:00 test.log.0005.heap
-rw-r--r-- 1 root root 1.0M Jun  1 15:00 test.log.0006.heap
```

- Analyze the heap files with pprof. There are two modes of analysis:
	- Complete mode. An analysis of the current heap is performed, showing some of the call paths for the current allocation of memory.

	```
	pprof --pdf python test.log.0012.heap
	```
	The command above will generate a file of profile00x.pdf, which can be opened directly, for example, [memory_cpu_allocator](https://github.com/jacquesqiao/Paddle/blob/bd2ea0e1f84bb6522a66d44a072598153634cade/doc/fluid/howto/optimization/memory_cpu_allocator.pdf).  As demonstrated in the chart below, during the running of the CPU version fluid, the module CPUAllocator is allocated with most memory. Other modules are allocated with relatively less memory, so they are ignored. It is very inconvenient for inspecting memory leak for memory leak is a chronic process which cannot be inspected in this picture.
	![result](https://user-images.githubusercontent.com/3048612/40964027-a54033e4-68dc-11e8-836a-144910c4bb8c.png)
	
	- Diff mode. You can do diff on the heap at two moments, which removes some modules whose memory allocation has not changed, and displays the incremental part.
	```
	pprof --pdf --base test.log.0010.heap python test.log.1045.heap
	```
	The generated result: [`memory_leak_protobuf`](https://github.com/jacquesqiao/Paddle/blob/bd2ea0e1f84bb6522a66d44a072598153634cade/doc/fluid/howto/optimization/memory_leak_protobuf.pdf)
	
	As shown from the figure: The structure of ProgramDesc has increased by 200MB+ between the two versions, so there is a large possibility that memory leak happens here, and the final result does prove a leak here.
	
	![result](https://user-images.githubusercontent.com/3048612/40964057-b434d5e4-68dc-11e8-894b-8ab62bcf26c2.png)
	![result](https://user-images.githubusercontent.com/3048612/40964063-b7dbee44-68dc-11e8-9719-da279f86477f.png)
	
