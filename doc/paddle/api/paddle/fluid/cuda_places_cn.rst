.. _cn_api_fluid_cuda_places:

cuda_places
-------------------------------

.. py:function:: paddle.fluid.cuda_places(device_ids=None)




.. note::
    多卡任务请先使用 FLAGS_selected_gpus 环境变量设置可见的GPU设备，下个版本将会修正 CUDA_VISIBLE_DEVICES 环境变量无效的问题。

该接口根据 ``device_ids`` 创建一个或多个 ``fluid.CUDAPlace`` 对象，并返回所创建的对象列表。

如果 ``device_ids`` 为 ``None``，则首先检查 ``FLAGS_selected_gpus`` 标志。
例如： ``FLAGS_selected_gpus=0,1,2`` ，则返回的列表将为 ``[fluid.CUDAPlace(0), fluid.CUDAPlace(1), fluid.CUDAPlace(2)]``。
如果未设置标志 ``FLAGS_selected_gpus`` ，则根据 ``CUDA_VISIBLE_DEVICES`` 环境变量，返回所有可见的 GPU places。

如果 ``device_ids`` 不是 ``None``，它应该是使用的GPU设备ID的列表或元组。
例如： ``device_id=[0,1,2]`` ，返回的列表将是 ``[fluid.CUDAPlace(0), fluid.CUDAPlace(1), fluid.CUDAPlace(2)]``。

参数：
  - **device_ids** (list(int)|tuple(int)，可选) - GPU的设备ID列表或元组。默认值为 ``None``。

返回: 创建的 ``fluid.CUDAPlace`` 列表。

返回类型：list[fluid.CUDAPlace]

**代码示例**

.. code-block:: python

      import paddle.fluid as fluid
      cuda_places = fluid.cuda_places()

