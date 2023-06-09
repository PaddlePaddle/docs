.. _cn_api_layers_ldexp:

ldexp
-------------------------------

.. py:function:: paddle.ldexp(x, y, name=None)

计算 `x` 乘以 2 的 `y` 次幂

.. math::
    out = x * 2^{y}

参数
::::::::::::

    - **x** (Tensor) - 多维 Tensor。数据类型为 float32、float64、int32、int64。
    - **y** (Tensor) - 多维 Tensor。通常为整数。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
输出为 Tensor，如果 ``x``， ``y`` 有不同的形状并且是可广播的，那么产生的张量形状是广播后x和y的形状。如果 ``x``， ``y`` 有相同的形状，其形状与 ``x``， ``y``  相同。数据类型是float32或float64。

代码示例
::::::::::::

COPY-FROM: paddle.ldexp
