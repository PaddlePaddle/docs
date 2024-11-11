.. _cn_api_paddle_index_add:

index_add
-------------------------------

.. py:function:: paddle.index_add(x, index, axis, value, name=None)



沿着指定轴 ``axis`` 将 ``index`` 中指定位置的 ``x`` 与 ``value`` 相加，并写入到结果 Tensor 中的对应位置。这里 ``index`` 是一个 ``1-D`` Tensor。除 ``axis`` 轴外，返回的 Tensor 其余维度大小和输入 ``x`` 相等， ``axis`` 维度的大小等于 ``index`` 的大小。

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

    * 示例 2 （输入为 2-D Tensor, axis=1 ）：

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
                        [1., 1., 1.],
                        [2., 1., 2.]]

** 示例 1 图解说明 **

    下图展示了示例 1 中的情形——一个形状为 [3,3] 的二维张量通过 index_add 操作在 axis=0 轴上对指定位置的元素进行相加，同时保持了除 ``axis`` 轴外，返回的 Tensor 其余维度大小和输入 ``x`` 相等。

    .. figure:: ../../images/api_legend/index_add/index_add-1.png
        :width: 500
        :alt: 示例 1 图示
        :align: center

**示例 2 图解说明**

    下图展示了示例 2 中的情形——一个形状为 [3,3] 的二维张量通过 index_add 操作在 axis=1 轴上对指定位置的元素进行相加，同时保持了除 ``axis`` 轴外，返回的 Tensor 其余维度大小和输入 ``x`` 相等。

    .. figure:: ../../images/api_legend/index_add/index_add-2.png
        :width: 500
        :alt: 示例 2 图示
        :align: center

参数
:::::::::

    - **x** （Tensor）– 输入 Tensor。 ``x`` 的数据类型可以是 float16, float32，float64，int32，int64。
    - **index** （Tensor）– 包含索引下标的 1-D Tensor。数据类型为 int32 或者 int64。
    - **axis**    (int) – 索引轴。数据类型为 int。
    - **value** （Tensor）– 与 ``x`` 相加的 Tensor。 ``value`` 的数据类型同 ``x`` 。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::

Tensor，返回一个数据类型同输入的 Tensor。


代码示例
::::::::::::

COPY-FROM: paddle.index_add
