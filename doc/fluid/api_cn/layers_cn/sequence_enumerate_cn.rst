.. _cn_api_fluid_layers_sequence_enumerate:

sequence_enumerate
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_enumerate(input, win_size, pad_value=0, name=None)

为输入索引序列生成一个新序列，该序列枚举输入长度为 ``win_size`` 的所有子序列。 输入序列和枚举序列第一维上维度相同，第二维是 ``win_size`` ，在生成中如果需要，通过设置 ``pad_value`` 填充。

**例子：**

::

        输入：
            X.lod = [[0, 3, 5]]  X.data = [[1], [2], [3], [4], [5]]  X.dims = [5, 1]
        属性：
            win_size = 2  pad_value = 0
        输出：
            Out.lod = [[0, 3, 5]]  Out.data = [[1, 2], [2, 3], [3, 0], [4, 5], [5, 0]]  Out.dims = [5, 2]

参数:
        - **input** （Variable）- 作为索引序列的输入变量。
        - **win_size** （int）- 枚举所有子序列的窗口大小。
        - **pad_value** （int）- 填充值，默认为0。

返回:      枚举序列变量是LoD张量（LoDTensor）。

返回类型:   Variable

**代码示例**

..  code-block:: python

      import paddle.fluid as fluid
      x = fluid.layers.data(shape[-1, 1], dtype='int32', lod_level=1)
      out = fluid.layers.sequence_enumerate(input=x, win_size=3, pad_value=0)









