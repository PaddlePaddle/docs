.. _cn_api_paddle_index_add:

index_add
-------------------------------

.. py:function:: paddle.index_add(x, index, axis, value, alpha=1, name=None, *, out=None)

.. note::

    本 API 支持两种签名：

    1. ``paddle.index_add(x, index, axis, value, alpha=1, name=None, *, out=None)`` （Paddle 风格）。
    2. ``paddle.index_add(input, dim, index, source, *, alpha=1, out=None)`` （PyTorch 风格）。


沿着指定轴 ``axis`` 将 ``index`` 中指定位置的 ``x`` 与 ``value`` 相加，并写入到结果 Tensor 中的对应位置。这里 ``index`` 是一个 ``1-D`` Tensor。返回 Tensor 与输入 ``x`` 的形状相同。

**示例**

::

    - 示例 1 （输入为 2-D Tensor, axis=0 ）：
        输入：
            x.shape = [3, 3]
            x.data = [[1., 1., 1.],
                      [1., 1., 1.],
                      [1., 1., 1.]]

        参数：
            index.shape = [2]
            index.data = [0, 2]

            axis = 0

            value.shape = [2, 3]
            value.data = [[1., 1., 1.],
                          [1., 1., 1.]]
        输出：
            out.shape = [3, 3]
            out.data = [[2., 2., 2.],
                        [1., 1., 1.],
                        [2., 2., 2.]]

    - 示例 2 （输入为 2-D Tensor, axis=1 ）：
        输入：
            x.shape = [3, 3]
            x.data = [[1., 1., 1.],
                      [1., 1., 1.],
                      [1., 1., 1.]]

        参数：
            index.shape = [2]
            index.data = [0, 2]

            axis = 1

            value.shape = [3, 2]
            value.data = [[1., 1.],
                          [1., 1.],
                          [1., 1.]]
        输出：
            out.shape = [3, 3]
            out.data = [[2., 1., 2.],
                        [2., 1., 2.],
                        [2., 1., 2.]]

**示例 1 图解说明**

    下图展示了示例 1 中的情形——一个形状为 [3,3] 的二维张量通过 index_add 操作在 axis=0 轴上对指定位置的元素进行相加，返回 Tensor 与输入 ``x`` 的形状相同。

    .. figure:: ../../images/api_legend/index_add/index_add-1.png
        :width: 500
        :alt: 示例 1 图示
        :align: center

**示例 2 图解说明**

    下图展示了示例 2 中的情形——一个形状为 [3,3] 的二维张量通过 index_add 操作在 axis=1 轴上对指定位置的元素进行相加，返回 Tensor 与输入 ``x`` 的形状相同。

    .. figure:: ../../images/api_legend/index_add/index_add-2.png
        :width: 500
        :alt: 示例 2 图示
        :align: center

参数
:::::::::

    - **x** (Tensor) - 输入 Tensor。 ``x`` 的数据类型可以是 float16，float32，float64，int32，int64。别名 ``input``。
    - **index** (Tensor) - 包含索引下标的 1-D Tensor。数据类型为 int32 或者 int64。
    - **axis** (int) - 索引轴。数据类型为 int。别名 ``dim``。
    - **value** (Tensor) - 与 ``x`` 相加的 Tensor。 ``value`` 的数据类型同 ``x``。别名 ``source``。
    - **alpha** (Number，可选) - ``value`` 的缩放因子。默认值为 1。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
::::::::::::

    - **out** (Tensor，可选) - 输出 Tensor。默认值为 None。

返回
:::::::::

Tensor，返回一个形状和数据类型均与输入 ``x`` 相同的 Tensor。


代码示例
::::::::::::

COPY-FROM: paddle.index_add
