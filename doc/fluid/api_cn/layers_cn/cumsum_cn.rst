.. _cn_api_fluid_layers_cumsum:

cumsum
-------------------------------

.. py:function:: paddle.fluid.layers.cumsum(x,axis=None,exclusive=None,reverse=None)

:alias_main: paddle.cumsum
:alias: paddle.cumsum,paddle.tensor.cumsum,paddle.tensor.math.cumsum
:old_api: paddle.fluid.layers.cumsum



沿给定轴(axis)的元素的累加和。默认结果的第一个元素和输入的第一个元素一致。如果exlusive为True，结果的第一个元素则为0。

参数：
    - **x** (Variable) - 累加的输入，需要进行累加操作的变量Tensor/LoDTensor。
    - **axis** (int，可选) - 指明需要累加的维。-1代表最后一维。默认为：-1。
    - **exclusive** (bool，可选) - 是否执行exclusive累加。默认为：False。
    - **reverse** (bool，可选) - 若为True，则以相反顺序执行累加。默认为：False。

返回：Variable(Tensor)。是累加的结果，即累加器的输出。

返回类型：变量(Variable)。

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(name="input", shape=[32, 784])
    result = fluid.layers.cumsum(data, axis=0)









