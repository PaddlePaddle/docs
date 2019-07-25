.. _cn_api_fluid_cpu_places:

cpu_places
-------------------------------

.. py:function:: paddle.fluid.cpu_places(device_count=None)

创建 ``fluid.CPUPlace`` 对象列表。

如果 ``device_count`` 为None，则设备数目将由环境变量 ``CPU_NUM`` 确定。如果未设置 ``CPU_NUM`` ，则设备数目默认为1，也就是说， ``CPU_NUM`` =1。

参数：
  - **device_count** (None|int) - 设备数目

返回: CPUPlace列表

返回类型：out (list(fluid.CPUPlace))

**代码示例**

.. code-block:: python

           import paddle.fluid as fluid
           cpu_places = fluid.cpu_places()


