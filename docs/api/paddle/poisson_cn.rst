.. _cn_api_tensor_poisson:

poisson
-------------------------------

.. py:function:: paddle.poisson(x, name=None)

该OP以输入 ``x`` 为泊松分布的 `lambda` 参数，生成一个泊松分布的随机数Tensor，输出Tensor的shape和dtype与输入Tensor相同。

.. math::
   out_i \sim Poisson(lambda = x_i)

参数
:::::::::
    - **x** (Tensor) - Tensor的每个元素，对应泊松分布的 ``lambda`` 参数。数据类型为： float32、float64。
    - **name** (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
:::::::::
`Tensor`：泊松分布的随机数Tensor，shape和dtype与输入 ``x`` 相同。


代码示例
:::::::::

.. code-block:: python

    import paddle
    paddle.set_device('cpu')
    paddle.seed(100)

    x = paddle.uniform([2,3], min=1.0, max=5.0)
    out = paddle.poisson(x)
    #[[2., 5., 0.],
    # [5., 1., 3.]]
