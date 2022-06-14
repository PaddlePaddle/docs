.. _cn_api_paddle_tensor_heaviside:

heaviside
-------------------------------

.. py:function:: paddle.heaviside(x, y, name=None)


逐元素地对 Tensor `x` 计算由 Tensor `y` 中的对应元素决定的赫维赛德阶跃函数，其计算公式为

.. math::
   \mathrm{heaviside}(x, y)=
      \left\{
            \begin{array}{lcl}
            0,& &\text{if } \ x < 0, \\
            y,& &\text{if } \ x = 0, \\
            1,& &\text{if } \ x > 0.
            \end{array}
      \right.

.. note::
   ``paddle.heaviside`` 遵守广播机制，如您想了解更多，请参见 :ref:`cn_user_guide_broadcasting`。

参数
:::::::::
   - **x** （Tensor）- 赫维赛德阶跃函数的输入 Tensor。数据类型为 float32、float64、int32 或 int64。
   - **y** （Tensor）- 决定了一个赫维赛德阶跃函数的 Tensor。数据类型为 float32、float64、int32 或 int64。
   - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
   `Tensor`，存储运算后的结果。如果 `x` 和 `y` 有不同的形状且是可以广播的，那么返回 Tensor 的形状是 `x` 和 `y` 经过广播后的形状。如果 `x` 和 `y` 有相同的形状，那么返回 Tensor 的形状与 `x` 和 `y` 相同。


代码示例
::::::::::
COPY-FROM: paddle.heaviside:heaviside-example
