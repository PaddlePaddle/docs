
数值计算
==================


enable_cublas_tensor_op_math
*******************************************
(始于1.2.0)

该flag表示是否使用Tensor Core，但可能会因此降低部分精确度。

取值范围
---------------
Bool型，缺省值为False。

示例
-------
enable_cublas_tensor_op_math=True - 使用Tensor Core。


use_mkldnn
*******************************************
(始于0.13.0)

在预测或训练过程中，可以通过该选项选择使用Intel MKL-DNN（https://github.com/intel/mkl-dnn）库运行。
“用于深度神经网络的英特尔（R）数学核心库（Intel(R) MKL-DNN）”是一个用于深度学习应用程序的开源性能库。该库加速了英特尔（R）架构上的深度学习应用程序和框架。Intel MKL-DNN包含矢量化和线程化构建建块，您可以使用它们来实现具有C和C ++接口的深度神经网络（DNN）。

取值范围
---------------
Bool型，缺省值为False。

示例
-------
FLAGS_use_mkldnn=True - 开启使用MKL-DNN运行。

注意
-------
FLAGS_use_mkldnn仅用于python训练和预测脚本。要在CAPI中启用MKL-DNN，请设置选项 -DWITH_MKLDNN=ON。
英特尔MKL-DNN支持英特尔64架构和兼容架构。
该库对基于以下设备的系统进行了优化：
英特尔SSE4.1支持的英特尔凌动（R）处理器；
第4代，第5代，第6代，第7代和第8代英特尔（R）Core（TM）处理器；
英特尔（R）Xeon（R）处理器E3，E5和E7系列（原Sandy Bridge，Ivy Bridge，Haswell和Broadwell）；
英特尔（R）Xeon（R）可扩展处理器（原Skylake和Cascade Lake）；
英特尔（R）Xeon Phi（TM）处理器（原Knights Landing and Knights Mill）；
兼容处理器。