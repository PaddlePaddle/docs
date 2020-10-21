.. _cn_api_tensor_bernoulli:

bernoulli
-------------------------------

.. py:function:: paddle.bernoulli(x, name=None)




该OP以输入 ``x`` 为概率，生成一个伯努利分布（0-1分布）的Tensor，输出Tensor的形状和数据类型与输入 ``x`` 相同。

.. math::
   out_i \sim Bernoulli(p = x_i)

参数：
    - **x** (Tensor) - 输入的概率值。数据类型为 ``float32`` 、``float64`` .
    - **name** (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回：
    Tensor：伯努利分布的随机Tensor，形状和数据类型为与输入 ``x`` 相同。


**代码示例**：

.. code-block:: python

    import paddle

    paddle.seed(100) # on CPU device
    x = paddle.rand([2,3])
    print(x.numpy())
    # [[0.5535528  0.20714243 0.01162981]
    # [0.51577556 0.36369765 0.2609165 ]]

    paddle.seed(200) # on CPU device
    out = paddle.bernoulli(x)
    print(out.numpy())
    # [[0. 0. 0.]
    # [1. 1. 0.]]








