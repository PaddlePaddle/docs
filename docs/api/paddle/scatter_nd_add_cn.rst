.. _cn_api_paddle_scatter_nd_add:

scatter_nd_add
-------------------------------

.. py:function:: paddle.scatter_nd_add(x, index, updates, name=None)




通过对 Tensor 中的单个值或切片应用稀疏加法，从而得到输出的 Tensor。

:code:`x` 是维度为 :code:`R` 的 Tensor。:code:`index` 是维度为 :code:`K` 的 Tensor。因此，:code:`index` 的形状是 :math:`[i_0, i_1, ..., i_{K-2}, Q]`，其中 :math:`Q \leq R` 。:code:`updates` 是一个维度为 :math:`K - 1 + R - Q` 的 Tensor，它的形状是 :math:`index.shape[:-1] + x.shape[index.shape[-1]:]` 。

根据 :code:`index` 的 :math:`[i_0, i_1, ..., i_{K-2}]` 得到相应的 :code:`updates` 切片，将其加到根据 :code:`index` 的最后一维得到 :code:`x` 切片上，从而得到最终的输出 Tensor。


示例：

::

        - 案例 1:
            x = [0, 1, 2, 3, 4, 5]
            index = [[1], [2], [3], [1]]
            updates = [9, 10, 11, 12]

          得到：

            output = [0, 22, 12, 14, 4, 5]

        - 案例 2:
            x = [[65, 17], [-14, -25]]
            index = [[], []]
            updates = [[[-1, -2], [1, 2]],
                       [[3, 4], [-3, -4]]]
            x.shape = (2, 2)
            index.shape = (2, 0)
            updates.shape = (2, 2, 2)

          得到：

            output = [[67, 19], [-16, -27]]

**示例一图解说明**：

    在这个示例中，通过 Paddle 的 scatter_nd_add 函数对张量 x 进行稀疏加法操作。初始张量 x 为 [0, 1, 2, 3, 4, 5]，通过 index 指定需要更新的索引位置，并使用 updates 中的值进行累加。scatter_nd_add 函数会根据 index 的位置逐步累加 updates 中的对应值，最终得到输出张量为 [0, 22, 12, 14, 4, 5]，实现了对张量部分元素的累加更新而保持其他元素不变。

参数
::::::::::::

    - **x** (Tensor) - 输入 Tensor，数据类型可以是 int32，int64，float32，float64。
    - **index** (Tensor) - 输入的索引 Tensor，数据类型为非负 int32 或非负 int64。它的维度 :code:`index.ndim` 必须大于 1，并且 :code:`index.shape[-1] <= x.ndim`
    - **updates** (Tensor) - 输入的更新 Tensor，它必须和 :code:`x` 有相同的数据类型。形状必须是 :code:`index.shape[:-1] + x.shape[index.shape[-1]:]` 。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，数据类型和形状都与 :code:`x` 相同。

代码示例
::::::::::::

COPY-FROM: paddle.scatter_nd_add

    .. figure:: ../../images/api_legend/scatter_nd_add.png
       :width: 700
       :alt: 示例一图示
       :align: center
