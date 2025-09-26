.. _cn_api_paddle_msort:

msort
-------------------------------

.. py:function:: paddle.msort(input: Tensor, *, out: Tensor | None = None)

沿输入 `Tensor` 的第 0 轴（`axis=0`）按升序对元素进行排序。

该函数等价于 `paddle.sort(x, axis=0)`。

参数
::::::::::::

    - **input** (Tensor) - 输入的 N-D  ``Tensor`` ，支持的数据类型为：float32、float64、int16、int32、int64、uint8。别名  ``input`` 。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
:::::::::

    - **out** (Tensor，可选) - 输出 Tensor，若不为  ``None`` ，计算结果将保存在该 Tensor 中，默认值为  ``None`` 。

返回
::::::::::::
Tensor，排序后的输出（与  ``input``  维度相同、数据类型相同）。


代码示例
::::::::::::

COPY-FROM: paddle.msort
