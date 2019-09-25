.. _cn_api_fluid_layers_pad:

pad
-------------------------------

.. py:function:: paddle.fluid.layers.pad(x, paddings, pad_value=0.0, name=None)

该OP在Tensor上填充一个由 ``pad_value`` 给出的常数值，填充宽度由 ``paddings`` 指定。
其中，维度 ``i`` 中 ``x`` 内容前填充的值个数用 ``paddings[2*i]`` 表示，维度 ``i`` 中 ``x`` 内容后填充的值个数用 ``paddings[2*i+1]`` 表示。

**样例**：

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
    - **x** (Variable) — 多维Tensor，数据类型为float32
    - **paddings** (list of integers) — 整数列表，指定每个维度填充值的个数。维度 ``i`` 中 ``x`` 内容前填充的值个数用 ``paddings[2*i]`` 表示，维度 ``i`` 中 ``x`` 内容后填充的值个数用 ``paddings[2*i+1]`` 表示。 ``paddings`` 长度必须是 ``rank(x)×2``
    - **pad_value** (float32, 可选) — 用来填充的常量值，数据类型为float。默认值为0.
    - **name** (str|None) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

返回： 填充后的Tensor，数据类型与输入 ``x`` 相同

返回类型： Variable


**代码示例**

..  code-block:: python

    # x 为一个秩为2的张量
    import paddle.fluid as fluid
    x = fluid.layers.data(name='data', shape=[224], dtype='float32')
    out = fluid.layers.pad(x=x, paddings=[0, 1, 1, 2], pad_value=0.)










