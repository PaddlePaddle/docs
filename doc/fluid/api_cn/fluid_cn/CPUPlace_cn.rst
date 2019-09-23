.. _cn_api_fluid_CPUPlace:

CPUPlace
-------------------------------

.. py:class:: paddle.fluid.CPUPlace

``CPUPlace`` 是一个设备描述符，表示一个分配或将要分配 ``Tensor`` 或 ``LoDTensor`` 的 ``CPU`` 设备，可以访问 ``CPUPlace`` 对应的内存。

**代码示例**

.. code-block:: python

        import paddle.fluid as fluid
        cpu_place = fluid.CPUPlace()


