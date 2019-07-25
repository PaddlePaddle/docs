.. _cn_api_fluid_layers_pad:

pad
-------------------------------

.. py:function:: paddle.fluid.layers.pad(x, paddings, pad_value=0.0, name=None)

在张量上加上一个由 ``pad_value`` 给出的常数值，填充宽度由 ``paddings`` 指定。
其中，维度 ``i`` 中 ``x`` 内容前填充的值个数用 ``paddings[i]`` 表示，维度 ``i`` 中 ``x`` 内容后填充的值个数用 ``paddings[i+1]`` 表示。

一个例子:

::

        Given:

         x = [[1, 2], [3, 4]]

        paddings = [0, 1, 1, 2]

        pad_value = 0

        Return:

        out = [[0, 1, 2, 0, 0]
               [0, 3, 4, 0, 0]
               [0, 0, 0, 0, 0]]


参数:
    - **x** (Variable) — —输入张量变量。
    - **paddings** (list) — 一个整数列表。按顺序填充在每个维度上填充元素。 ``padding`` 长度必须是 ``rank(x)×2``
    - **pad_value** (float) — 用来填充的常量值。
    - **name** (str|None) — 这个层的名称(可选)。如果设置为None，该层将被自动命名。

返回： 填充后的张量变量

返回类型： 变量（Variable）


**代码示例**

..  code-block:: python

    # x 为一个秩为2的张量
    import paddle.fluid as fluid
    x = fluid.layers.data(name='data', shape=[224], dtype='float32')
    out = fluid.layers.pad(
        x=x, paddings=[0, 1, 1, 2], pad_value=0.)










