##########
性能调优
##########

介绍 Fluid 使用过程中的调优方法，包括：

	  - `如何进行基准测试 <benchmark.html>`_：介绍如何选择基准模型，从而验证模型的精度和性能
	  - `CPU性能调优 <cpu_profiling_cn.html>`_：介绍如何使用 cProfile 包、yep库、Google perftools 进行性能分析与调优
	  - `GPU性能调优 <gpu_profiling_cn.html>`_：介绍如何使用 Fluid 内置的定时工具、nvprof 或 nvvp 进行性能分析和调优
	  - `堆内存分析和优化 <host_memory_profiling_cn.html>`_：介绍如何使用 gperftool 进行堆内存分析和优化，以解决内存泄漏的问题
	  - `Timeline工具简介 <timeline_cn.html>`_ ：介绍如何使用 Timeline 工具进行性能分析和调优

..  toctree::
	:hidden:

    benchmark.rst
    cpu_profiling_cn.md
    gpu_profiling_cn.rst
    host_memory_profiling_cn.md
    timeline_cn.md
