.. _cn_api_fluid_layers_pad:

pad
-------------------------------

.. py:function:: paddle.fluid.layers.pad(x, paddings, pad_value=0.0, name=None)




该 OP 在 Tensor 上填充一个由 ``pad_value`` 给出的常数值，填充宽度由 ``paddings`` 指定。
其中，维度 ``i`` 中 ``x`` 内容前填充的值个数用 ``paddings[2*i]`` 表示，维度 ``i`` 中 ``x`` 内容后填充的值个数用 ``paddings[2*i+1]`` 表示。

**示例**：

.. code-block:: text

        Given:
            x = [[1, 2], [3, 4]]

            paddings = [0, 1, 1, 2]

            pad_value = 0

        Return:
            out = [[0, 1, 2, 0, 0]
                   [0, 3, 4, 0, 0]
                   [0, 0, 0, 0, 0]]


参数
::::::::::::

    - **x** (Variable) — 多维 Tensor，数据类型为 float32
    - **paddings** (list of integers) — 整数列表，指定每个维度填充值的个数。维度 ``i`` 中 ``x`` 内容前填充的值个数用 ``paddings[2*i]`` 表示，维度 ``i`` 中 ``x`` 内容后填充的值个数用 ``paddings[2*i+1]`` 表示。``paddings`` 长度必须是 ``rank(x)×2``
    - **pad_value** (float32，可选) — 用来填充的常量值，数据类型为 float。默认值为 0。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 填充后的 Tensor，数据类型与输入 ``x`` 相同

返回类型
::::::::::::
 Variable


代码示例
::::::::::::

..  code-block:: python

    # x 为一个秩为 2 的 Tensor
    import paddle.fluid as fluid
    x = fluid.data(name='data', shape=[300, 300], dtype='float32')
    out = fluid.layers.pad(x=x, paddings=[0, 1, 1, 2], pad_value=0.)
