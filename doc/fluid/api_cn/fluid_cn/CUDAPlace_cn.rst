.. _cn_api_fluid_CUDAPlace:

CUDAPlace
-------------------------------

.. py:class:: paddle.fluid.CUDAPlace

``CUDAPlace`` 是一个设备描述符，表示一个分配或将要分配 ``Tensor`` 或 ``LoDTensor`` 的 ``GPU`` 设备，并且每个 ``CUDAPlace`` 有一个 ``dev_id``（设备id）来表明当前 ``CUDAPlace`` 代表的显卡数。
``dev_id`` 不同的 ``CUDAPlace`` 所对应的内存不可相互访问。

参数：
  - **id** (int，可选) - GPU的设备ID。如果为 ``None``，则默认会使用 id 为 0 的设备。缺省值为 ``None``。

**代码示例**

.. code-block:: python

       import paddle.fluid as fluid
       gpu_place = fluid.CUDAPlace(0)




