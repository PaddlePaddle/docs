.. _cn_api_paddle_tensor_i0e:

i0e
-------------------------------

.. py:function:: paddle.i0e(x, name=None)


对于给定 ``x`` 计算其每个元素的第一类指数缩放零阶修正贝塞尔曲线函数，其中输入 ``x`` 大小无特殊限制。返回一个第一类指数缩放零阶修正贝塞尔曲线函数上的 Tensor。

.. math::
    I_0(x)=\sum^{\infty}_{k=0}\frac{(x^2/4)^k}{(k!)^2} \\
    I_{0e}(x)=e^{-\lvert x\rvert}I_0(x)

参数
::::::::::
    - **x** (Tensor) – 输入是一个多维的 Tensor，它的数据类型可以是 float32，float64。
    - **name** (str，可选) - 具体用法请参见  :ref:`api_guide_Name` ，一般无需设置，默认值为 None。
返回
::::::::::
    - ``Tensor`` (Tensor)：在 x 处的第一类指数缩放零阶修正贝塞尔曲线函数的值。


代码示例
::::::::::

COPY-FROM: paddle.i0e
