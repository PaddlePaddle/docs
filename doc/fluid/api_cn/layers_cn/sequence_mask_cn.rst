.. _cn_api_fluid_layers_sequence_mask:

sequence_mask
-------------------------------

.. py:function::  paddle.fluid.layers.sequence_mask(x, maxlen=None, dtype='int64', name=None)

该层根据输入 ``x`` 和 ``maxlen`` 输出一个掩码，数据类型为 ``dtype`` 。

假设 x 是一个形状为 ``[d_1, d_2，…, d_n]`` 的张量， 则输出 y 是一个形为 ``[d_1, d_2，… ，d_n, maxlen]`` 的掩码，其中:

.. math::

  y(i_1, i_2,..., i_n, j) = (j < x(i_1, i_2,..., i_n))

范例如下：

::

    给定输入：
      x = [3, 1, 1, 0]  maxlen = 4

    得到输出张量：
      mask = [[1, 1, 1, 0],
              [1, 0, 0, 0],
              [1, 0, 0, 0],
              [0, 0, 0, 0]]
        




参数：
  - **x** (Variable) - 输入张量，其元素是小于等于 ``maxlen`` 的整数，Tensor或LoDTensor。
  - **maxlen** (int，可选) - 序列的最大长度。默认为空，此时 ``maxlen`` 取 ``x`` 中所有元素的最大值。
  - **dtype** (np.dtype|core.VarDesc.VarType|str，可选) - 输出的数据类型，默认为 ``int64`` 。
  - **name** (str，可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。

返回： mask张量，LoDTensor。

返回类型： Variable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.layers as layers
    
    x = fluid.layers.data(name='x', shape=[10], dtype='float32', lod_level=1)
    mask = layers.sequence_mask(x=x)










