.. _cn_api_fluid_cuda_places:

cuda_places
-------------------------------

.. py:function:: paddle.fluid.cuda_places(device_ids=None)

创建 ``fluid.CUDAPlace`` 对象列表。



如果 ``device_ids`` 为None，则首先检查 ``FLAGS_selected_gpus`` 的环境变量。如果 ``FLAGS_selected_gpus=0,1,2`` ，则返回的列表将为[fluid.CUDAPlace(0), fluid.CUDAPlace(1), fluid.CUDAPlace(2)]。如果未设置标志 ``FLAGS_selected_gpus`` ，则将返回所有可见的GPU places。


如果 ``device_ids`` 不是None，它应该是GPU的设备ID。例如，如果 ``device_id=[0,1,2]`` ，返回的列表将是[fluid.CUDAPlace(0), fluid.CUDAPlace(1), fluid.CUDAPlace(2)]。

参数：
  - **device_ids** (None|list(int)|tuple(int)) - GPU的设备ID列表

返回: CUDAPlace列表

返回类型：out (list(fluid.CUDAPlace))

**代码示例**

.. code-block:: python

      import paddle.fluid as fluid
      cuda_places = fluid.cuda_places()

