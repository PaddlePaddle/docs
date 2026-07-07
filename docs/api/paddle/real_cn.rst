.. _cn_api_paddle_real:

real
------

.. py:function:: paddle.real(x, name=None, *, out=None)

返回一个包含输入复数 Tensor 的实部数值的新 Tensor。

参数
::::::::::::

    - **x** (Tensor) - 输入 Tensor，其数据类型可以为 complex64 或 complex128。别名 ``input``。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
::::::::::::
    - **out** (Tensor，可选) - 输出 Tensor，默认值为 None。

返回
::::::::::::
Tensor，包含原复数 Tensor 的实部数值。

代码示例
::::::::::::

COPY-FROM: paddle.real
