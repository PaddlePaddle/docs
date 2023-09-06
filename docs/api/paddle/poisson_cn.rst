.. _cn_api_tensor_poisson:

poisson
-------------------------------

.. py:function:: paddle.poisson(x, name=None)

以输入参数 ``x`` 为泊松分布的 `lambda` 参数，生成一个泊松分布的随机数 Tensor，输出 Tensor 的 shape 和 dtype 与输入 Tensor 相同。

.. math::
   out_i \sim Poisson(lambda = x_i)

参数
:::::::::
    - **x** (Tensor) - Tensor 的每个元素，对应泊松分布的 ``lambda`` 参数。数据类型为：bfloat16, float16, float32、float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
`Tensor`：泊松分布的随机数 Tensor，shape 和 dtype 与输入 ``x`` 相同。


代码示例
:::::::::

COPY-FROM: paddle.poisson
