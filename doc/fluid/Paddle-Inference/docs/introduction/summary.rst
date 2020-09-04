概述
========

Paddle Inference为飞桨核心框架推理引擎。Paddle Inference功能特性丰富，性能优异，针对不同平台不同的应用场景进行了深度的适配优化,做到高吞吐、低时延，保证了飞桨模型在服务器端即训即用，快速部署。    

特性
-------

- 通用性。支持对Paddle训练出的所有模型进行预测。

- 内存/显存复用。在推理初始化阶段，对模型中的OP输出Tensor 进行依赖分析，将两两互不依赖的Tensor在内存/显存空间上进行复用，进而增大计算并行量，提升服务吞吐量。


- 细粒度OP融合。在推理初始化阶段，按照已有的融合模式将模型中的多个OP融合成一个OP，减少了模型的计算量的同时，也减少了 Kernel Launch的次数，从而能提升推理性能。目前Paddle Inference支持的融合模式多达几十个。


- 高性能CPU/GPU Kernel。内置同Intel、Nvidia共同打造的高性能kernel，保证了模型推理高性能的执行。


- 子图集成 `TensorRT <https://developer.nvidia.com/tensorrt>`_。Paddle Inference采用子图的形式集成TensorRT，针对GPU推理场景，TensorRT可对一些子图进行优化，包括OP的横向和纵向融合，过滤冗余的OP，并为OP自动选择最优的kernel，加快推理速度。


- 集成MKLDNN
   
- 支持加载PaddleSlim量化压缩后的模型。 `PaddleSlim <https://github.com/PaddlePaddle/PaddleSlim>`_ 是飞桨深度学习模型压缩工具，Paddle Inference可联动PaddleSlim，支持加载量化、裁剪和蒸馏后的模型并部署，由此减小模型存储空间、减少计算占用内存、加快模型推理速度。其中在模型量化方面，`Paddle Inference在X86 CPU上做了深度优化 <https://github.com/PaddlePaddle/PaddleSlim/tree/80c9fab3f419880dd19ca6ea30e0f46a2fedf6b3/demo/mkldnn_quant/quant_aware>`_ ，常见分类模型的单线程性能可提升近3倍，ERNIE模型的单线程性能可提升2.68倍。
	
支持系统及硬件   
------------

支持服务器端X86 CPU、NVIDIA GPU芯片，兼容Linux/macOS/Windows系统。     

同时也支持NVIDIA Jetson嵌入式平台。

语言支持
------------

- 支持Pyhton语言
- 支持C++ 语言 
- 支持Go语言 
- 支持R语言  
	
**下一步**

- 如果您刚接触Paddle Inference， 请访问 `Quick start <./quick_start.html>`_。
