.. _cn_api_paddle_diagflat:

diagflat
-------------------------------

.. py:function:: paddle.diagflat(x, offset=0, name=None)


如果 ``x`` 是一维 Tensor，则返回带有 ``x`` 元素作为对角线的二维方阵。

如果 ``x`` 是大于等于二维的 Tensor，则返回一个二维方阵，其对角线元素为 ``x`` 在连续维度展开得到的一维 Tensor 的元素。

参数 ``offset`` 控制对角线偏移量：

- 如果 ``offset`` = 0，则为主对角线。
- 如果 ``offset`` > 0，则为上对角线。
- 如果 ``offset`` < 0，则为下对角线。

参数
:::::::::
    - **x** (Tensor) - 输入的 `Tensor`。它的形状可以是任意维度。其数据类型应为 float16，float32，float64，int32，int64。
    - **offset** (int，可选) - 对角线偏移量。正值表示上对角线，0 表示主对角线，负值表示下对角线。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
``Tensor``，方阵。输出数据类型与输入数据类型相同。


代码示例 1
:::::::::

COPY-FROM: paddle.diagflat:diagflat-example-1

代码示例 2
:::::::::

COPY-FROM: paddle.diagflat:diagflat-example-2
