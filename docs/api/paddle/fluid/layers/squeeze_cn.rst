.. _cn_api_fluid_layers_squeeze:

squeeze
-------------------------------

.. py:function:: paddle.fluid.layers.squeeze(input, axes, name=None)




该 OP 会根据 axes 压缩输入 Tensor 的维度。如果指定了 axes，则会删除 axes 中指定的维度，axes 指定的维度要等于 1。如果没有指定 axes，那么所有等于 1 的维度都会被删除。

- 例 1：

.. code-block:: python

        输入：
            X.shape = [1,3,1,5]
            axes = [0]
        输出；
            Out.shape = [3,1,5]
- 例 2：

.. code-block:: python

        输入：
            X.shape = [1,3,1,5]
            axes = []
        输出：
            Out.shape = [3,5]
- 例 3：

.. code-block:: python

        输入：
            X.shape = [1,3,1,5]
            axes = [-2]
        输出：
            Out.shape = [1,3,5]

参数
::::::::::::

        - **input** (Variable) - 输入任意维度的 Tensor。支持的数据类型：float32，float64，int8，int32，int64。
        - **axes** (list) - 输入一个或一列整数，代表要压缩的轴。axes 的范围：:math:`[-rank(input), rank(input))` 。 axes 为负数时，:math:`axes=axes+rank(input)` 。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 返回对维度进行压缩后的 Tensor。数据类型与输入 Tensor 一致。

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
