.. _cn_api_tensor_repeat_interleave:

repeat_interleave
-------------------------------

.. py:function:: paddle.repeat_interleave(x, repeats, axis=None, name=None)


沿着指定轴 ``axis`` 对输入 ``x`` 进行复制，创建并返回到一个新的 Tensor。当 ``repeats`` 为 ``1-D`` Tensor 时，``repeats``  长度必须和指定轴 ``axis`` 维度一致，``repeats`` 对应位置的值表示 ``x`` 对应位置元素需要复制的次数。当 ``repeats`` 为 int 时，``x`` 沿指定轴 ``axis`` 上所有元素复制 ``repeats`` 次。

参数
:::::::::
    - **x** （Tensor）– 输入 Tensor。 ``x`` 的数据类型可以是 float32，float64，int32，int64。
    - **repeats** （Tensor, int）– 包含复制次数的 1-D Tensor 或指定的复制次数。
    - **axis**    (int，可选) – 指定对输入 ``x`` 进行运算的轴，若未指定，默认值为 None，使用输入 Tensor 的 flatten 形式。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
:::::::::
    - **Tensor**：返回一个数据类型同输入的 Tensor。

代码示例
:::::::::

COPY-FROM: paddle.repeat_interleave
