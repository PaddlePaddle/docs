.. _cn_api_paddle_diag:

diag
-------------------------------

.. py:function:: paddle.diag(x, offset=0, padding_value=0, name=None)


如果 ``x`` 是向量（1-D Tensor），则返回带有 ``x`` 元素作为对角线的 2-D 方阵。

如果 ``x`` 是矩阵（2-D Tensor），则提取 ``x`` 的对角线元素，以 1-D Tensor 返回。

参数 ``offset`` 控制对角线偏移量：

- 如果 ``offset`` = 0，则为主对角线。
- 如果 ``offset`` > 0，则为上对角线。
- 如果 ``offset`` < 0，则为下对角线。

参数
:::::::::
    - **x** (Tensor) - 输入的 `Tensor`。它的形状可以是一维或二维。其数据类型应为 float16、float32、float64、int32、int64。
    - **offset** (int，可选) - 对角线偏移量。正值表示上对角线，0 表示主对角线，负值表示下对角线。默认值为 0。
    - **padding_value** (int|float，可选) -使用此值来填充指定对角线以外的区域。仅在输入为一维 Tensor 时生效。默认值为 0。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
``Tensor``，方阵或向量。输出数据类型与输入数据类型相同。


代码示例 1
:::::::::

COPY-FROM: paddle.diag:code-example-1

代码示例 2
:::::::::

COPY-FROM: paddle.diag:code-example-2
