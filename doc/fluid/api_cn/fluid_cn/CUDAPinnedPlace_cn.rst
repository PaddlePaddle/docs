.. _cn_api_fluid_CUDAPinnedPlace:

CUDAPinnedPlace
-------------------------------

.. py:class:: paddle.fluid.CUDAPinnedPlace

``CUDAPinnedPlace`` 是一个设备描述符，它所指代的存储空间可以转移数据到 GPU 中，可以被 GPU 和 CPU 访问。
有关 CUDA 的数据转移和 ``pinned memory``，参见 `官方文档 <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#pinned-memory>`_ 。

**代码示例**

.. code-block:: python

      import paddle.fluid as fluid
      place = fluid.CUDAPinnedPlace()

