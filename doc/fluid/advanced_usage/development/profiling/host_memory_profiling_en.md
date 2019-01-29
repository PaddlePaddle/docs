# Analysis and Optimization of Heap

Every computer program may have the potential of memory leak. As for **Memory Leak**, generally, the program allocates memory on the heap without releasing it. As the memory of the program becomes larger and larger, it will affect the stability of the program, which may make the running speed slower or cause oom(Out of Memory ), even affecting the stability of the machine running the program, causing downtime.


There are many memory leak analysis tools at present. Typically, [valgrind](http://valgrind.org/docs/manual/quick-start.html#quick-start.intro), [gperftools](https://gperftools.github.io/gperftools/).

Because Fluid is run in Python-driven C++ core, valgrind is very difficult to analyze directly. You need to compile the debug version of the dedicated Python version with valgrind support, and most of the output information is Python's own symbols and call information. It's very difficult. In addition, using valgrind will make the program run very slowly, so it is not recommended.

The documents mainly introduces the use of [gperftools](https://gperftools.github.io/gperftools/) .

gperftool mainly supports such four functions:

- thread-caching malloc
- heap-checking using tcmalloc
- heap-profiling using tcmalloc
- CPU profiler

Paddle also provides a [tutorial on CPU performance analysis](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/howto/optimization/cpu_profiling_cn.md) based on gperftool.

For the analysis of heap, we mainly use thread-caching malloc and heap-profiling using tcmalloc.

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

- Use heap profile to run python program. Essentially, a snapshot of the heap allocation is done periodically.

```
# HEAPPROFILE sets the directory and file prefix of the generated heap analysis file
# HEAP_PROFILE_ALLOCATION_INTERVAL Sets how many storage dumps are allocated for each dump, default 1GB
env HEAPPROFILE="./perf_log/test.log" HEAP_PROFILE_ALLOCATION_INTERVAL=209715200 python trainer.py
```

As the program runs, a lot of files are generated in the perf_log folder as follows:

```
-rw-r--r-- 1 root root 1.0M Jun  1 15:00 test.log.0001.heap
-rw-r--r-- 1 root root 1.0M Jun  1 15:00 test.log.0002.heap
-rw-r--r-- 1 root root 1.0M Jun  1 15:00 test.log.0003.heap
-rw-r--r-- 1 root root 1.0M Jun  1 15:00 test.log.0004.heap
-rw-r--r-- 1 root root 1.0M Jun  1 15:00 test.log.0005.heap
-rw-r--r-- 1 root root 1.0M Jun  1 15:00 test.log.0006.heap
```

- Analyze the heap file with pprof. There are two modes of analysis:
	- Complete mode. An analysis of the current heap is performed, showing some of the call paths for the current allocation of memory.

	```
	pprof --pdf python test.log.0012.heap
	```
	The command above will generate a file of profile00x.pdf, which can be opened directly, for example, [memory_cpu_allocator](https://github.com/jacquesqiao/Paddle/blob/bd2ea0e1f84bb6522a66d44a072598153634cade/doc/fluid/howto/optimization/memory_cpu_allocator.pdf). As can be seen from the figure below, during the running of the CPU version fluid, the most stored modular CPUAllocator is allocated. Other modules are relatively less allocated memory, so they are ignored, which is very inconvenient for allocating memory leaks. The leak is a slow process which cannot be seen in this picture.
	![result](https://user-images.githubusercontent.com/3048612/40964027-a54033e4-68dc-11e8-836a-144910c4bb8c.png)
	
	- Diff mode. You can do diff on the heap at two times, removing some modules whose memory allocation has not changed, and displaying the incremental part.
	```
	pprof --pdf --base test.log.0010.heap python test.log.1045.heap
	```
	The generated result: [`memory_leak_protobuf`](https://github.com/jacquesqiao/Paddle/blob/bd2ea0e1f84bb6522a66d44a072598153634cade/doc/fluid/howto/optimization/memory_leak_protobuf.pdf)
	
	As can be seen from the figure: The structure of ProgramDesc has increased by 200MB+ between the two versions, so there is a large possibility of memory leaks here, and the final result does prove to be a leak here.
	
	![result](https://user-images.githubusercontent.com/3048612/40964057-b434d5e4-68dc-11e8-894b-8ab62bcf26c2.png)
	![result](https://user-images.githubusercontent.com/3048612/40964063-b7dbee44-68dc-11e8-9719-da279f86477f.png)
	
