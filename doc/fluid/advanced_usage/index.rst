########
进阶使用
########

=====================
        概览
=====================
..  todo::

如果您非常熟悉 Fluid，期望获得更高效的模型或者定义自己的Operator，请阅读：

	- `移动端部署 <../advanced_usage/deploy/index_mobile.html>`_：介绍了 PaddlePaddle 组织下的嵌入式平台深度学习框架——Paddle-Mobile，包括：

	- `简介 <../advanced_usage/deploy/mobile_readme.html>`_：简要介绍了 Paddle-Mobile 的应用效果，特点以及使用说明
	- `环境搭建 <../advanced_usage/deploy/mobile_build.html>`_：从使用 Docker 和不使用 Docker 两种方法下分别介绍如何搭建环境
	- `ios开发文档 <../advanced_usage/deploy/mobile_dev.html>`_：介绍如何在 ios 系统下运用 Paddle-Mobile 进行开发

	- `Anakin预测引擎 <../advanced_usage/deploy/index_anakin.html>`_：介绍如何使用 Anakin 在不同硬件平台实现深度学习的高速预测

	- `如何写新的Operator <../advanced_usage/development/new_op.html>`_ ：介绍如何在 Fluid 中添加新的 Operator

	- `Op相关的一些注意事项 <../advanced_usage/development/op_notes.html>`_ ：介绍Op相关的一些注意事项

	- `性能调优 <../advanced_usage/development/profiling/index.html>`_ ：介绍 Fluid 使用过程中的调优方法，包括：

	  - `如何进行基准测试 <../advanced_usage/development/profiling/benchmark.html>`_：介绍如何选择基准模型，从而验证模型的精度和性能
	  - `CPU性能调优 <../advanced_usage/development/profiling/cpu_profiling_cn.html>`_：介绍如何使用 cProfile 包、yep库、Google perftools 进行性能分析与调优
	  - `GPU性能调优 <../advanced_usage/development/profiling/gpu_profiling_cn.html>`_：介绍如何使用 Fluid 内置的定时工具、nvprof 或 nvvp 进行性能分析和调优
	  - `堆内存分析和优化 <../advanced_usage/development/profiling/host_memory_profiling_cn.html>`_：介绍如何使用 gperftool 进行堆内存分析和优化，以解决内存泄漏的问题
	  - `Timeline工具简介 <../advanced_usage/development/profiling/timeline_cn.html>`_ ：介绍如何使用 Timeline 工具进行性能分析和调优


非常欢迎您为我们的开源社区做出贡献，关于如何贡献您的代码或文档，请阅读：

	- `如何贡献代码 <../advanced_usage/development/contribute_to_paddle.html>`_：介绍如何向 PaddlePaddle 开源社区贡献代码

	- `如何贡献文档 <../advanced_usage/development/write_docs_cn.html>`_：介绍如何向 PaddlePaddle 开源社区贡献文档

=====================
        目录
=====================

..  toctree::
    :maxdepth: 2

    deploy/index_mobile.rst
    deploy/index_anakin.rst
    development/contribute_to_paddle/index_cn.rst
    development/write_docs_cn.md
    development/new_op.md
    development/op_notes.md
    development/profiling/index.rst
