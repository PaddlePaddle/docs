.. _cn_api_fluid_layers_sequence_expand:

sequence_expand
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_expand(x, y, ref_level=-1, name=None)

序列扩张层（Sequence Expand Layer)

将根据指定 y 的 level lod 展开输入变量x，请注意 x 的 lod level 最多为1，而 x 的秩最少为2。当 x 的秩大于2时，它就被看作是一个二维张量。下面的例子将解释 sequence_expand 是如何工作的:

::


    * 例1
      x is a LoDTensor:
    x.lod  = [[2,        2]]
    x.data = [[a], [b], [c], [d]]
    x.dims = [4, 1]

      y is a LoDTensor:
    y.lod = [[2,    2],
             [3, 3, 1, 1]]

      ref_level: 0

      then output is a 1-level LoDTensor:
    out.lod =  [[2,        2,        2,        2]]
    out.data = [[a], [b], [a], [b], [c], [d], [c], [d]]
    out.dims = [8, 1]

    * 例2
      x is a Tensor:
    x.data = [[a], [b], [c]]
    x.dims = [3, 1]

      y is a LoDTensor:
    y.lod = [[2, 0, 3]]

      ref_level: -1

      then output is a Tensor:
    out.data = [[a], [a], [c], [c], [c]]
    out.dims = [5, 1]

参数：
    - **x** (Variable) - 输入变量，张量或LoDTensor
    - **y** (Variable) - 输入变量，为LoDTensor
    - **ref_level** (int) - x表示的y的Lod层。若设为-1，表示lod的最后一层
    - **name** (str|None) - 该层名称（可选）。如果设为空，则自动为该层命名

返回：扩展变量，LoDTensor

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.layers as layers
    x = fluid.layers.data(name='x', shape=[10], dtype='float32')
    y = fluid.layers.data(name='y', shape=[10, 20],
                 dtype='float32', lod_level=1)
    out = fluid.layers.sequence_expand(x=x, y=y, ref_level=0)









