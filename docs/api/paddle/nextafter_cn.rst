.. _cn_api_paddle_nextafter:

nextafter
-------------------------------

.. py:function:: paddle.nextafter(x, y, name=None)




逐元素将 x 之后的下一个浮点值返回给 y。

输入 `x` 和 `y` 的形状（shape）必须是可广播的（broadcastable）。详情请参考 `Tensor 的广播机制 <../../guides/beginner/tensor_cn.html#id7>`_ 。

参数
:::::::::
    - **x** (Tensor) - 输入的 Tensor，数据类型为：float32、float64。
    - **y** (Tensor) - 输入的 Tensor，数据类型为：float32、float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
输出 Tensor，与 ``x`` 维度相同、数据类型相同。

代码示例
:::::::::

COPY-FROM: paddle.nextafter
