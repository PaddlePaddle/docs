.. _cn_api_fluid_layers_cumsum:

cumsum
-------------------------------

.. py:function:: paddle.fluid.layers.cumsum(x,axis=None,exclusive=None,reverse=None)

沿给定轴的元素的累加和。默认结果的第一个元素和输入的第一个元素一致。如果exlusive为真，结果的第一个元素则为0。

参数：
    - **x** -累加操作符的输入
    - **axis** (INT)-需要累加的维。-1代表最后一维。[默认 -1]。
    - **exclusive** (BOOLEAN)-是否执行exclusive累加。[默认false]。
    - **reverse** (BOOLEAN)-若为true,则以相反顺序执行累加。[默认 false]。

返回：累加器的输出

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(name="input", shape=[32, 784])
    result = fluid.layers.cumsum(data, axis=0)









