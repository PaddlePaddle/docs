.. _cn_api_fluid_layers_squeeze:

squeeze
-------------------------------

.. py:function:: paddle.fluid.layers.squeeze(input, axes, name=None)




该OP会根据axes压缩输入Tensor的维度。如果指定了axes，则会删除axes中指定的维度，axes指定的维度要等于1。如果没有指定axes，那么所有等于1的维度都会被删除。

- 例1：

.. code-block:: python

        输入：
            X.shape = [1,3,1,5]
            axes = [0]
        输出；
            Out.shape = [3,1,5]
- 例2：

.. code-block:: python

        输入：
            X.shape = [1,3,1,5]
            axes = []
        输出：
            Out.shape = [3,5]
- 例3：

.. code-block:: python

        输入：
            X.shape = [1,3,1,5]
            axes = [-2]
        输出：
            Out.shape = [1,3,5]

参数
::::::::::::

        - **input** (Variable) - 输入任意维度的Tensor。支持的数据类型：float32，float64，int8，int32，int64。
        - **axes** (list) - 输入一个或一列整数，代表要压缩的轴。axes的范围：:math:`[-rank(input), rank(input))` 。 axes为负数时，:math:`axes=axes+rank(input)` 。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 返回对维度进行压缩后的Tensor。数据类型与输入Tensor一致。

返回类型
::::::::::::
Variable

代码示例
::::::::::::

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.layers as layers
    x = layers.data(name='x', shape=[5, 1, 10])
    y = layers.squeeze(input=x, axes=[1]) #y.shape=[5, 10]









