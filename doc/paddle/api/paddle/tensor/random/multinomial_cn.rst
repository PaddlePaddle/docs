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

    x = paddle.rand([2,4])
    print(x.numpy())
    # [[0.7713825  0.4055941  0.433339   0.70706886]
    # [0.9223313  0.8519825  0.04574518 0.16560672]]

    out1 = paddle.multinomial(x, num_samples=5, replacement=True)
    print(out1.numpy())
    # [[3, 3, 1, 1, 0]
    # [0, 0, 0, 0, 1]]

    # out2 = paddle.multinomial(x, num_samples=5)
    # OutOfRangeError: When replacement is False, number of samples
    #  should be less than non-zero categories

    out3 = paddle.multinomial(x, num_samples=3)
    print(out3.numpy())
    # [[0, 2, 3]
    # [0, 1, 3]]








