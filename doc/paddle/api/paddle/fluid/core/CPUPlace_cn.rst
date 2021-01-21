.. _cn_api_fluid_CPUPlace:

CPUPlace
-------------------------------

.. py:class:: paddle.fluid.CPUPlace




``CPUPlace`` 是一个设备描述符，指定 ``CPUPlace`` 则 ``Tensor`` 将被自动分配在该设备上，并且模型将会运行在该设备上。

**代码示例**

.. code-block:: python

        import paddle
        cpu_place = paddle.CPUPlace()


