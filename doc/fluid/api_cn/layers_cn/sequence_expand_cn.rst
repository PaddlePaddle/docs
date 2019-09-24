.. _cn_api_fluid_layers_sequence_expand:

sequence_expand
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_expand(x, y, ref_level=-1, name=None)

序列扩张层（Sequence Expand Layer)，根据输入 ``y`` 的第 ``ref_level`` 层lod对输入 ``x`` 进行扩展。注意 ``x`` 的lod level最多为1，而 ``x`` 的秩最少为2。当 ``x`` 的秩大于2时，将被当作是一个二维张量处理。

范例如下：

::

    例1：
    给定输入一维LoDTensor x：
      x.lod  = [[2,        2]]
      x.data = [[a], [b], [c], [d]]
      x.dims = [4, 1]
    和输入 y：
      y.lod = [[2,    2],
              [3, 3, 1, 1]]
    指定 ref_level = 0

    输出为1级LoDTensor：
      out.lod =  [[2,        2,        2,        2]]
      out.data = [[a], [b], [a], [b], [c], [d], [c], [d]]
      out.dims = [8, 1]

::

    例2：
    给定输入一维LoDTensor x：
      x.data = [[a], [b], [c]]
      x.dims = [3, 1]
    和输入 y：
      y.lod = [[2, 0, 3]]
    默认 ref_level = -1

    输出为1级LoDTensor：
      out.data = [[a], [a], [c], [c], [c]]
      out.dims = [5, 1]

参数：
    - **x** (Variable) - 输入变量，Tensor或LoDTensor。
    - **y** (Variable) - 输入变量，LoDTensor。
    - **ref_level** (int，可选) - 扩展 ``x`` 所依据的 ``y`` 的lod层。默认值-1，表示lod的最后一层。
    - **name** (str，可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。

返回：扩展变量，LoDTensor

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.layers as layers
    x = fluid.layers.data(name='x', shape=[10], dtype='float32')
    y = fluid.layers.data(name='y', shape=[10, 20],
                 dtype='float32', lod_level=1)
    out = layers.sequence_expand(x=x, y=y, ref_level=0)









