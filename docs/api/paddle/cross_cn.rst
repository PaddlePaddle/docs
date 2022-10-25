.. _cn_api_tensor_linalg_cross:

cross
-------------------------------

.. py:function:: paddle.cross(x, y, axis=None, name=None)


计算张量 ``x`` 和 ``y`` 在 ``axis`` 维度上的向量积（叉积）。

``x`` 和 ``y`` 必须有相同的形状，且指定的 ``axis`` 的长度必须为 3。如果未指定 ``axis``，默认选取第一个长度为 3 的 ``axis`` 。

参数
:::::::::
    - **x** (Tensor) - 第一个输入张量。
    - **y** (Tensor) - 第二个输入张量。
    - **axis** (int，可选) – 沿着此维进行向量积操作。默认值是 9，意思是选取第一个长度为 3 的 ``axis`` 。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
``Tensor``，向量积的结果。

代码示例
::::::::::

COPY-FROM: paddle.cross
