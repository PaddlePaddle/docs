.. _cn_api_paddle_geometric_:

geometric\_
-------------------------------

.. py:function:: paddle.geometric_(x, probs, name=None)
使用从几何分布中抽取的数据填充 Tensor。

参数
::::::::::::

    - **x** (Tensor) - 被填充的 Tensor ，数据类型为： float32、float64。
    - **probs** (float | Tensor) - 输入的概率参数， probs 的值必须为正，数据类型为 float、Tensor。当该参数类型为 Tensor 时， probs 代表每次试验成功的概率。
    - **name** (str ，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
填充了服从几何分布随机数的输入 Tensor。

代码示例
::::::::::::

COPY-FROM: paddle.geometric_
