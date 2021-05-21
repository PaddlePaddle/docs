.. _cn_api_fluid_cuda_pinned_places:

cuda_pinned_places
-------------------------------


.. py:function:: paddle.fluid.cuda_pinned_places(device_count=None)






该接口创建 ``device_count`` 个 ``fluid.CUDAPinnedPlace`` ( fluid. :ref:`cn_api_fluid_CUDAPinnedPlace` ) 对象，并返回所创建的对象列表。

如果 ``device_count`` 为 ``None``，实际设备数目将由当前任务中使用的GPU设备数决定。用户可通过以下2种方式设置任务可用的GPU设备：

- 设置环境变量 ``FLAGS_selected_gpus`` ，例如 ``export FLAGS_selected_gpus='0,1'``。
- 设置环境变量 ``CUDA_VISIBLE_DEVICES`` ，例如 ``export CUDA_VISIBLE_DEVICES='0,1'``。

关于如何设置任务中使用的GPU设备，具体请查看 fluid. :ref:`cn_api_fluid_cuda_places`  。

参数：
  - **device_count** (int，可选) - 设备数目。默认值为 ``None``。

返回: ``fluid.CUDAPinnedPlace`` 对象列表。

返回类型：list[fluid.CUDAPinnedPlace]

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    # 1）不设置任何环境变量，默认使用所有的GPU，8卡的机器则将创建8个CUDAPinnedPlace
    # 2）export FLAGS_selected_gpus='0,1'，则创建2个CUDAPinnedPlace
    cuda_pinned_places = fluid.cuda_pinned_places()
    # 3）创建1个CUDAPinnedPlace
    cuda_pinned_places = fluid.cuda_pinned_places(1)
