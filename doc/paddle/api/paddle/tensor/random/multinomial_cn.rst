.. _cn_api_tensor_multinomial:

multinomial
-------------------------------

.. py:function:: paddle.multinomial(x, num_samples=1, replacement=False, name=None)




该OP以输入 ``x`` 为概率，生成一个多项分布的Tensor。
输入 ``x`` 是用来随机采样的概率分布， ``x`` 中每个元素都应该大于等于0，且不能都为0。
参数 ``replacement`` 表示它是否是一个可放回的采样，如果 ``replacement`` 为True, 能重复对一种类别采样。

参数：
    - **x** (Tensor) - 输入的概率值。数据类型为 ``float32`` 、``float64`` .
    - **num_samples** (int, 可选) - 采样的次数（可选，默认值为1）。
    - **replacement** (bool, 可选) - 是否是可放回的采样（可选，默认值为False）。
    - **name** (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回：
    Tensor：多项分布采样得到的随机Tensor，为 ``num_samples`` 次采样得到的类别下标。


**代码示例**：

.. code-block:: python

    import paddle

    paddle.manual_seed(100) # on CPU device
    x = paddle.rand([2,4])
    print(x.numpy())
    # [[0.5535528  0.20714243 0.01162981 0.51577556]
    # [0.36369765 0.2609165  0.18905126 0.5621971 ]]

    paddle.manual_seed(200) # on CPU device
    out1 = paddle.multinomial(x, num_samples=5, replacement=True)
    print(out1.numpy())
    # [[3 3 0 0 0]
    # [3 3 3 1 0]]

    # out2 = paddle.multinomial(x, num_samples=5)
    # InvalidArgumentError: When replacement is False, number of samples
    #  should be less than non-zero categories

    paddle.manual_seed(300) # on CPU device
    out3 = paddle.multinomial(x, num_samples=3)
    print(out3.numpy())
    # [[3 0 1]
    # [3 1 0]]








