.. _cn_api_paddle_CUDAPinnedPlace:

CUDAPinnedPlace
-------------------------------

.. py:class:: paddle.CUDAPinnedPlace




``CUDAPinnedPlace`` 是一个设备描述符，它所指代的页锁定内存由 CUDA 函数 ``cudaHostAlloc()`` 在主机内存上分配，主机的操作系统将不会对这块内存进行分页和交换操作，可以通过直接内存访问技术访问，加速主机和 GPU 之间的数据拷贝。
有关 CUDA 的数据转移和 ``pinned memory``，参见 `官方文档 <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#pinned-memory>`_ 。

代码示例
::::::::::::

COPY-FROM: paddle.CUDAPinnedPlace
