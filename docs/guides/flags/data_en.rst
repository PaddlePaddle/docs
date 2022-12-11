
data processing
==================

FLAGS_enable_cublas_tensor_op_math
*******************************************
(since 1.2.0)

This Flag indicates whether to use Tensor Core, but it may lose some precision.

Values accepted
---------------
Bool. The default value is False.

Example
-------
FLAGS_enable_cublas_tensor_op_math=True will use Tensor Core.


FLAGS_use_mkldnn
*******************************************
(since 0.13.0)

Give a choice to run with Intel MKL-DNN (https://github.com/intel/mkl-dnn) library on inference or training.

Intel(R) Math Kernel Library for Deep Neural Networks (Intel(R) MKL-DNN) is an open-source performance library for deep-learning applications. The library accelerates deep-learning applications and frameworks on Intel(R) architecture. Intel MKL-DNN contains vectorized and threaded building blocks that you can use to implement deep neural networks (DNN) with C and C++ interfaces.

Values accepted
---------------
Bool. The default value is False.

Example
-------
FLAGS_use_mkldnn=True will enable running with MKL-DNN support.

Note
-------
FLAGS_use_mkldnn is only used for python training and inference scripts. To enable MKL-DNN in CAPI, set build option -DWITH_MKLDNN=ON
Intel MKL-DNN supports Intel 64 architecture and compatible architectures. The library is optimized for the systems based on:
Intel Atom(R) processor with Intel SSE4.1 support
4th, 5th, 6th, 7th, and 8th generation Intel(R) Core(TM) processor
Intel(R) Xeon(R) processor E3, E5, and E7 family (formerly Sandy Bridge, Ivy Bridge, Haswell, and Broadwell)
Intel(R) Xeon(R) Scalable processors (formerly Skylake and Cascade Lake)
Intel(R) Xeon Phi(TM) processors (formerly Knights Landing and Knights Mill)
and compatible processors.
