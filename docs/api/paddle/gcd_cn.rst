.. _cn_api_paddle_tensor_gcd:

gcd
-------------------------------

.. py:function:: paddle.gcd(x, y, name=None)

计算两个输入的按元素绝对值的最大公约数，输入必须是整型。

.. note::

    gcd(0,0)=0, gcd(0, y)=|y|

    如果 x 和 y 的 shape 不一致，会对两个 shape 进行广播操作，得到一致的 shape（并作为输出结果的 shape），
    请参见 `Tensor 介绍`_ .

    .. _Tensor 介绍: ../../guides/beginner/tensor_cn.html#id7

参数
:::::::::

- **x**  (Tensor) - 输入的 Tensor，数据类型为：int32，int64。
- **y**  (Tensor) - 输入的 Tensor，数据类型为：int32，int64。
- **name**  (str，可选） - 操作的名称(可选，默认值为 None）。更多信息请参见 :ref:`api_guide_Name`。

返回
:::::::::

输出 Tensor，与输入数据类型相同。

代码示例
:::::::::

COPY-FROM: paddle.gcd
