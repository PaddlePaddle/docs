.. _cn_api_paddle_tensor_i1e:

i1e
-------------------------------

.. py:function:: paddle.i1e(x, name=None)
对于给定 ``x`` 计算其每个元素的第一类指数缩放的一阶修正贝塞尔曲线函数，其中输入 ``x`` 大小无特殊限制。返回第一类指数缩放的一阶修正贝塞尔函数对应输出 Tensor。

.. math::
    I_1e(x)=\exp (-|x|) * i 1(x)=\exp (-|x|) * \frac{\left(\text { input }_{i}\right)}{2} * \sum_{k=0}^{\infty} \frac{\left(\text { input }_{i}^{2} / 4\right)^{k}}{(k !) *(k+1) !}

参数
::::::::::
    - **x** (Tensor) – 输入是一个多维的 Tensor，它的数据类型可以是 float32，float64。
    - **name** (str，可选) - 具体用法请参见  :ref:`api_guide_Name` ，一般无需设置，默认值为 None。

返回
::::::::::
    - ``Tensor`` (Tensor)：在 x 处的第一类指数缩放的一阶修正贝塞尔曲线函数的值。


代码示例
::::::::::

COPY-FROM: paddle.i1e
