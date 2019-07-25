.. _cn_api_fluid_layers_squeeze:

squeeze
-------------------------------

.. py:function:: paddle.fluid.layers.squeeze(input, axes, name=None)

向张量维度中移除单维输入。传入用于压缩的轴。如果未提供轴，所有的单一维度将从维中移除。如果选择的轴的形状条目不等于1，则报错。

::


    例如：

    例1：
        给定
            X.shape = (1,3,1,5)
            axes = [0]
        得到
            Out.shape = (3,1,5)
    例2：
        给定
            X.shape = (1,3,1,5)
            axes = []
        得到
            Out.shape = (3,5)

参数：
        - **input** (Variable)-将要压缩的输入变量
        - **axes** (list)-一列整数，代表压缩的维
        - **name** (str|None)-该层名称

返回：输出压缩的变量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.layers as layers
    x = fluid.layers.data(name='x', shape=[5, 1, 10])
    y = fluid.layers.sequeeze(input=x, axes=[1])









