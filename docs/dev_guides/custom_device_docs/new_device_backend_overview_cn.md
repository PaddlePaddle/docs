# 硬件接入飞桨后端方案介绍

随着芯片多样化的发展，不同芯片的硬件架构、指令集的设计可能不同，其软件栈及适用场景也存在差异，因此硬件与深度学习框架间存在多样化的接入方案。即使是同一款芯片也存在不同硬件接入方案，比如 Nvidia GPU 提供了 CUDA （ToolKit）高级编程接口可直接支持算子开发、高性能的 cuDNN 加速库（SDK）的接入方式、适用于推理场景的图引擎接入方式 TensorRT（SDK）；一些硬件支持深度学习编译器方案，比如 TVM、MLIR；一些硬件支持神经网络交换格式方案 ONNX。

不同硬件可能存在多种技术方案并行，方案成熟度也各不相同，因此为了满足各种硬件厂商的接入需求，也为了匹配芯片中长期技术发展的规划，飞桨提供了多种硬件接入方案，包括算子开发与映射、子图与整图接入、深度学习编译器后端接入以及开源神经网络格式转化四种硬件接入方案，供硬件厂商根据自身芯片架构设计与软件栈的建设成熟度进行灵活选择。

> 说明：硬件厂商如需接入飞桨深度学习框架，可参考本文指导，想咨询更多信息，可发邮件至：Paddle-better@baidu.com。

<center><img src="https://github.com/PaddlePaddle/docs/blob/develop/docs/dev_guides/custom_device_docs/images/new_device_backend_overview_cn.png?raw=true" width="900" ></center>

<center>图 1 硬件接入飞桨后端的整体方案<center/>

| **硬件接入方案**<img width=900/> | **方案介绍** | **适用场景**     |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 算子开发与映射方案 | 该方案以算子为粒度对接，即对硬件侧算子库与飞桨框架侧算子库做映射，其优势是调度灵活，支持上层应用开发者以动态图的编程范式进行模型开发和调试，开发体验更佳。<ul><li>Built-in 方案：侵入式硬件接入方案。设备管理层和算子库等嵌入到飞桨框架中，与飞桨一起编译打包使用，开发算子代码需遵循飞桨相关开发规范。飞桨最早适配的 CPU、GPU 等硬件采用了该方案，新接入飞桨的硬件不推荐采用该方案。</li><li> [Plugin](./custom_device_overview_cn.html) 方案：可解耦插件式硬件接入方案（也叫 Custom Device 方案）。飞桨提供自定义 Runtime、自定义 Kernel 与自定义 CCL 标准 C/C++接口，支持硬件厂商分别接入其软件栈的 Driver/Runtime、DNN 算子库与集合通信库。适配过程中开发者无需关心飞桨框架底层概念，只需基于上述接口实现对应的功能即可。</li></ul>其中，对于算子 Kernel 的开发，飞桨还推出了 [Kernel Primitive API](../op_optimization/kernel_primitive_api/index_cn.html) 方案：一种更高效的 Kernel 开发方案，目前已实现一部分 API，还在持续扩充中。对算子 Kernel 实现中的底层代码进行了抽象与功能封装，提供高性能的 Block 级 IO 运算和 Compute 运算。使用该方案进行 Kernel 开发可以更加专注计算逻辑的实现，在保证性能的同时大幅减少代码量，同时实现了算子计算与硬件解耦。 | <ul><li>适用于 CPU 、GPU 这类可提供通用编程语言/指令集、数学库和算子库的硬件，如 Nvidia GPU 可提供 cuDNN SDK、CUDA 高级编程语言，提供类似技术路线的硬件即可采用该方案。</li><li>支持接入飞桨训练框架和 Paddle Inference 原生推理框架。</li><li>建议新硬件接入飞桨框架采用更便捷的 Plugin 方案接入，如 [华为昇腾 NPU](https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/npu/README_cn.md)、[寒武纪 MLU](https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/mlu/README_cn.md) 均采用了该方式。</li></ul> |
| 子图与整图接入方案   | 该方案主要面向推理场景，以整图/子图为粒度对接，即实现硬件侧神经网络中间表示（IR）与飞桨框架侧神经网络中间表示的转换。该方案的优势是所有硬件相关的细节如模型的优化、生成和执行均在硬件图引擎内实现，给予硬件厂商较大的灵活性，同时能够降低硬件适配深度学习框架难度和人力成本。<br/>飞桨支持了整图/子图 Built-in 接入方案，也提供了可解耦插件式接入的 [NNAdapter](https://www.paddlepaddle.org.cn/lite/develop/develop_guides/nnadapter.html#paddle-lite) 方案，实现了推理框架和硬件适配解耦，包括设备管理、多设备统一上下文管理、模型组网、模型编译与生成以及模型执行等标准适配接口，硬件厂商仅需完成这些标准接口实现即可快速接入飞桨框架。 | <ul><li>Built-in 方案适用于 Nvidia GPU TensorRT 的推理接入，采用该方式接入的硬件可支持在 Paddle Inference 上使用。</li><li>[NNAdapter](https://www.paddlepaddle.org.cn/lite/develop/develop_guides/nnadapter.html#paddle-lite) 方案适用于提供了推理专用的图引擎的新型 AI 硬件的适配，如华为昇腾 NPU、寒武纪 MLU、高通 QNN 等，可支持在飞桨轻量化推理引擎 Paddle Lite 上使用。</li></ul> |
| 编译器后端接入方案   | 该方案通过“中间表示代码生成方式”（Codegen）来接入硬件。编译器分为前端和后端，前端对接各种网络模型并将其转化为代表计算图的高层中间表示（HLIR），后端进一步转化计算图中间表示为计算指令级别的低层中间表示（LLIR），并基于此代码生成器（Codegen）产生在硬件上运行的指令。在各层中间表示（IR）上，编译器都会进行相关优化。硬件厂商只需对神经网络编译器的底层 IR 开发相应的代码生成器（Codegen），即可通过更低层次的方式（指从 CINN IR 转化为机器指令的过程）生成在具体硬件上的可执行指令（Machine Code）。<br/>飞桨支持了开源深度学习编译器 [TVM](https://github.com/apache/tvm/)，也提供了自研 [CINN](https://github.com/PaddlePaddle/CINN)（Compiler Infrastructure for Neural Networks）编译器，目前该自研方案正在探索和研发中。 | <ul><li>适用于支持底层汇编指令的硬件。</li><li>支持通过 CINN 编译器接入飞桨训练框架和 Paddle Inference 原生推理框架。</li></ul> |
| 神经网络格式转化方案 | 该方案面向推理场景，飞桨框架支持将飞桨模型格式转化到开源的 ONNX 模型格式。很多在线或离线的推理引擎都支持 ONNX 开源格式，这样飞桨模型也可以通过这些推理引擎部署到更多的硬件类型上。<br/>飞桨提供了模型转换工具 [Paddle2ONNX](../../guides/advanced/model_to_onnx_cn.html)，将飞桨模型转换为 ONNX 格式。 | <ul><li>适用于支持 ONNX 格式模型部署的硬件。</li></ul>       |
