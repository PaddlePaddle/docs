.. _cn_api_nn_cn_glu:

glu
-------------------------------

.. py:function:: paddle.nn.functional.glu(x, axis=-1, name=None)

门控线性单元。输入按照给定的维度二等分，其中第一部分被用作内容，第二部分经过一个 sigmoid 函数
之后被用作门限。输入是内容和门限的逐元素乘积。

.. math::
    \mathrm{GLU}(a, b) = a \otimes \sigma(b)

参数:
::::::::::
 - x (Tensor) - 输入的 ``Tensor`` ，数据类型为：float32、float64。
 - axis (int, 可选) - 沿着该轴将输入二等分。D 为输入的维数，则 ``axis`` 应该在 [-D, D) 的范围内。如 ``axis`` 为负数，则相当于 ``axis +D``.
 - name (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
::::::::::
    ``Tensor`` ，数据类型同 ``x`` 一致，在指定的轴上其尺寸减半。

代码示例
::::::::::
COPY-FROM: paddle.nn.functional.glu
