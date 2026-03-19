.. _cn_api_paddle_cummax:

cummax
-------------------------------

.. py:function:: paddle.cummax(x, axis=None, dtype='int64', name=None, *, out=None)

沿给定 ``axis`` 计算 Tensor ``x`` 的累积最大值。

本 API 支持两种调用方式：

1. **Paddle 风格**： ``paddle.cummax(x, axis=None, dtype='int64', name=None)``
   ``axis`` 参数可选，默认为 ``None``。

2. **PyTorch 风格**： ``paddle.cummax(input, dim, *, out=None)``
   ``dim`` 参数必选。参数名 ``input`` 是 ``x`` 的别名，``dim`` 是 ``axis`` 的别名。

.. note::
    结果的第一个元素和输入的第一个元素相同。

参数
::::::::::
    - **x** (Tensor) - 需要进行累积最大值操作的 Tensor。别名 ``input``。
    - **axis** (int，可选) - 指明需要累积最大值的维度。-1 代表最后一维。默认：None，将输入展开为一维变量再进行累积最大值计算。别名 ``dim``。
    - **dtype** (str|paddle.dtype|np.dtype，可选) - 输出 Indices 的数据类型，可以是 int32、int64，默认值为 int64。
    - **name** (str，可选) - 具体用法请参见  :ref:`api_guide_Name` ，一般无需设置，默认值为 None。

关键字参数
::::::::::
    - **out** (tuple[Tensor, Tensor]，可选) - 输出 Tensor 元组，包含两个 Tensor (values, indices)。若不为 ``None``，计算结果将保存在该 Tensor 元组中，默认值为 ``None``。

返回
::::::::::
    - ``values`` (Tensor)：返回累积最大值操作的结果，累积最大值结果类型和输入 x 相同。
    - ``indices`` (Tensor)：返回对应累积最大值操作的索引结果。

代码示例
::::::::::

COPY-FROM: paddle.cummax
