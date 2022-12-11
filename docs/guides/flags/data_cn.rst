
数值计算
==================


FLAGS_enable_cublas_tensor_op_math
*******************************************
(始于 1.2.0)

该 flag 表示是否使用 Tensor Core，但可能会因此降低部分精确度。

取值范围
---------------
Bool 型，缺省值为 False。

示例
-------
FLAGS_enable_cublas_tensor_op_math=True - 使用 Tensor Core。


FLAGS_use_mkldnn
*******************************************
(始于 0.13.0)

在预测或训练过程中，可以通过该选项选择使用 Intel MKL-DNN（https://github.com/intel/mkl-dnn）库运行。
“用于深度神经网络的英特尔（R）数学核心库（Intel(R) MKL-DNN）”是一个用于深度学习应用程序的开源性能库。该库加速了英特尔（R）架构上的深度学习应用程序和框架。Intel MKL-DNN 包含矢量化和线程化构建建块，您可以使用它们来实现具有 C 和 C ++接口的深度神经网络（DNN）。

取值范围
---------------
Bool 型，缺省值为 False。

示例
-------
FLAGS_use_mkldnn=True - 开启使用 MKL-DNN 运行。

注意
-------
FLAGS_use_mkldnn 仅用于 python 训练和预测脚本。要在 CAPI 中启用 MKL-DNN，请设置选项 -DWITH_MKLDNN=ON。
英特尔 MKL-DNN 支持英特尔 64 架构和兼容架构。
该库对基于以下设备的系统进行了优化：
英特尔 SSE4.1 支持的英特尔凌动（R）处理器；
第 4 代，第 5 代，第 6 代，第 7 代和第 8 代英特尔（R）Core（TM）处理器；
英特尔（R）Xeon（R）处理器 E3，E5 和 E7 系列（原 Sandy Bridge，Ivy Bridge，Haswell 和 Broadwell）；
英特尔（R）Xeon（R）可扩展处理器（原 Skylake 和 Cascade Lake）；
英特尔（R）Xeon Phi（TM）处理器（原 Knights Landing and Knights Mill）；
兼容处理器。
