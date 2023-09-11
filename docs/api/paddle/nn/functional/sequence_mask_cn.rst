.. _cn_api_paddle_nn_functional_sequence_mask:

sequence_mask
-------------------------------

.. py:function::  paddle.nn.functional.sequence_mask(x, maxlen=None, dtype='int64', name=None)




该层根据输入 ``x`` 和 ``maxlen`` 输出一个掩码，数据类型为 ``dtype`` 。

假设 x 是一个形状为 ``[d_1, d_2，…, d_n]`` 的 Tensor，则输出 y 是一个形状为 ``[d_1, d_2，… ，d_n, maxlen]`` 的掩码，其中：

.. math::

  y(i_1, i_2,..., i_n, j) = (j < x(i_1, i_2,..., i_n))

范例如下：

::

    给定输入：
      x = [3, 1, 1, 0]  maxlen = 4

    得到输出 Tensor：
      mask = [[1, 1, 1, 0],
              [1, 0, 0, 0],
              [1, 0, 0, 0],
              [0, 0, 0, 0]]





参数
:::::::::
  - **x** (Tensor) - 输入 Tensor，其元素是小于等于 ``maxlen`` 的整数，形状为 ``[d_1, d_2，…, d_n]`` 的 Tensor。
  - **maxlen** (int，可选) - 序列的最大长度。默认为空，此时 ``maxlen`` 取 ``x`` 中所有元素的最大值。
  - **dtype** (np.dtype|core.VarDesc.VarType|str，可选) - 输出的数据类型，默认为 ``int64`` 。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
mask Tensor，Tensor，形状为 ``[d_1, d_2，… ，d_n, maxlen]``，数据类型由 ``dtype`` 指定，支持 float32、float64、int32 和 int64，默认为 int64。

代码示例
:::::::::
COPY-FROM: paddle.nn.functional.sequence_mask
