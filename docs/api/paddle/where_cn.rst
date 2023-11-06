.. _cn_api_paddle_where:

where
-------------------------------

.. py:function:: paddle.where(condition, x=None, y=None, name=None)




根据 ``condition`` 来选择 ``x`` 或 ``y`` 中的对应元素来组成新的 Tensor。具体地，

.. math::
    out_i =
    \begin{cases}
    x_i, & \text{if}  \ condition_i \  \text{is} \ True \\
    y_i, & \text{if}  \ condition_i \  \text{is} \ False \\
    \end{cases}.

.. note::
    ``numpy.where(condition)`` 功能与 ``paddle.nonzero(condition, as_tuple=True)`` 相同，可以参考 :ref:`cn_api_paddle_nonzero`。

参数
::::::::::::

    - **condition** (Tensor) - 选择 ``x`` 或 ``y`` 元素的条件。在为 True（非零值）时，选择 ``x``，否则选择 ``y``。
    - **x** (Tensor|scalar，可选) - 条件为 True 时选择的 Tensor 或 scalar，数据类型为 bfloat16、 float16、float32、float64、int32 或 int64。``x`` 和 ``y`` 必须都给出或者都不给出。
    - **y** (Tensor|scalar，可选) - 条件为 False 时选择的 Tensor 或 scalar，数据类型为 bfloat16、float16、float32、float64、int32 或 int64。``x`` 和 ``y`` 必须都给出或者都不给出。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，形状与 ``condition`` 相同，数据类型与 ``x`` 和 ``y`` 相同。



代码示例
::::::::::::
COPY-FROM: paddle.where
