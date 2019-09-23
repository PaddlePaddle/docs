.. _cn_api_fluid_cuda_pinned_places:

cuda_pinned_places
-------------------------------


.. py:function:: paddle.fluid.cuda_pinned_places(device_count=None)



该接口根据 ``device_count`` 创建一个或多个 ``fluid.CUDAPinnedPlace`` 对象，并返回所创建的对象列表。

如果 ``device_count`` 为 ``None``，则设备数目将由环境变量 ``CPU_NUM`` 确定。如果未设置 ``CPU_NUM`` ，则设备数目将由 ``multiprocessing.cpu_count()`` 确定。

参数：
  - **device_count** (int，可选) - 设备数目。缺省值：``None``。

返回: ``fluid.CUDAPinnedPlace`` 对象列表。

返回类型：list[fluid.CUDAPinnedPlace]

**代码示例**

.. code-block:: python

        import paddle.fluid as fluid
        cuda_pinned_places_cpu_num = fluid.cuda_pinned_places()
        # 或者
        cuda_pinned_places = fluid.cuda_pinned_places(1)

