.. _cn_api_paddle_i0e:

i0e
-------------------------------

.. py:function:: paddle.i0e(x, name=None, *, out=None)


对于给定 ``x`` 计算其每个元素的第一类指数缩放零阶修正贝塞尔曲线函数，其中输入 ``x`` 大小无特殊限制。返回一个第一类指数缩放零阶修正贝塞尔曲线函数上的 Tensor。

.. math::
    I_0(x)=\sum^{\infty}_{k=0}\frac{(x^2/4)^k}{(k!)^2} \\
    I_{0e}(x)=e^{-\lvert x\rvert}I_0(x)

参数
::::::::::
    - **x** (Tensor) – 输入是一个多维的 Tensor，它的数据类型可以是 float32、float64、uint8、int8、int16、int32、int64。别名 ``input``。
    - **name** (str，可选) - 具体用法请参见  :ref:`api_guide_Name` ，一般无需设置，默认值为 ``None``。

关键字参数
:::::::::
    - **out** (Tensor，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。

返回
::::::::::
    - ``Tensor`` (Tensor)：在 x 处的第一类指数缩放零阶修正贝塞尔曲线函数的值。整数类型输入会自动转换为 float32。


代码示例
::::::::::

COPY-FROM: paddle.i0e
