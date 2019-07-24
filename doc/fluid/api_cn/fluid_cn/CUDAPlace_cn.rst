.. _cn_api_fluid_CUDAPlace:

CUDAPlace
-------------------------------

.. py:class:: paddle.fluid.CUDAPlace

CUDAPlace是一个设备描述符，它代表一个GPU，并且每个CUDAPlace有一个dev_id（设备id）来表明当前CUDAPlace代表的卡数。dev_id不同的CUDAPlace所对应的内存不可相互访问。

**代码示例**

.. code-block:: python

       import paddle.fluid as fluid
       gpu_place = fluid.CUDAPlace(0)




