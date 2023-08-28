.. _cn_api_paddle_atan2:

atan2
-------------------------------

.. py:function:: paddle.atan2(x, y, name=None)




对 x/y 进行逐元素的 arctangent 运算，通过符号确定象限

.. math::
    atan2(x,y)=\left\{\begin{matrix}
    & tan^{-1}(\frac{x}{y}) & y > 0 \\
    & tan^{-1}(\frac{x}{y}) + \pi & x>=0, y < 0 \\
    & tan^{-1}(\frac{x}{y}) - \pi & x<0, y < 0 \\
    & +\frac{\pi}{2} & x>0, y = 0 \\
    & -\frac{\pi}{2} & x<0, y = 0 \\
    &\text{undefined} & x=0, y = 0
    \end{matrix}\right.

参数
:::::::::

- **x**  (Tensor) - 输入的 Tensor，数据类型为：int32、int64、float16、float32、float64。
- **y**  (Tensor) - 输入的 Tensor，数据类型为：int32、int64、float16、float32、float64。
- **name**  (str，可选) - 操作的名称（可选，默认值为 None）。更多信息请参见 :ref:`api_guide_Name`。

返回
:::::::::

输出 Tensor，与 ``x`` 维度相同、数据类型相同（输入为 int 时，输出数据类型为 float64）。

代码示例
:::::::::

COPY-FROM: paddle.atan2
