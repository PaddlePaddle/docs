.. _cn_api_paddle_true_divide_:

true_divide_
-------------------------------

.. py:function:: paddle.true_divide_(x, y, name=None)

该 API 是 ``true_divide`` 的 inplace 版本，对输入 Tensor 进行原地除法操作。

参数
::::::::::::

    - **x** (Tensor) - 输入的 Tensor，会被原地修改。别名 ``input``。
    - **y** (Tensor) - 输入的 Tensor，作为除数。别名 ``other``。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，与 ``x`` 是同一个 Tensor，包含逐元素除法后的结果。

代码示例
::::::::::::

COPY-FROM: paddle.true_divide_
