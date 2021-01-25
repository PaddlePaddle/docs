.. _cn_api_nn_functional_nll_loss:

nll_loss
-------------------------------
.. py:function:: paddle.nn.functional.nll_loss(input, label, weight=None, ignore_index=-100, reduction='mean', name=None)

该接口返回 `negative log likelihood` 。可在 :ref:`cn_api_nn_loss_NLLLoss` 查看详情。

参数
:::::::::
    - **input** (Tensor): - 输入 `Tensor`, 其形状为 :math:`[N, C]` , 其中 `C` 为类别数。但是对于多维度的情形下，它的形状为 :math:`[N, C, d_1, d_2, ..., d_K]` 。数据类型为float32或float64。
    - **label** (Tensor): - 输入x对应的标签值。其形状为 :math:`[N,]` 或者 :math:`[N, d_1, d_2, ..., d_K]`, 数据类型为int64。
    - **weight** (Tensor, 可选): - 手动指定每个类别的权重。其默认为 `None` 。如果提供该参数的话，长度必须为 `num_classes` 。数据类型为float32或float64。
    - **ignore_index** (int64, 可选): - 指定一个忽略的标签值，此标签值不参与计算。默认值为-100。数据类型为int64。
    - **reduction** (str, 可选): - 指定应用于输出结果的计算方式，可选值有: `none`, `mean`, `sum` 。默认为 `mean` ，计算 `mini-batch` loss均值。设置为 `sum` 时，计算 `mini-batch` loss的总和。设置为 `none` 时，则返回loss Tensor。数据类型为string。
    - **name** (str, 可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

返回
:::::::::
`Tensor` ，返回存储表示 `negative log likelihood loss` 的损失值。

代码示例
:::::::::

.. code-block:: python

        import paddle
        from paddle.nn.functional import nll_loss
        log_softmax = paddle.nn.LogSoftmax(axis=1)
        
        input = paddle.to_tensor([[0.88103855, 0.9908683 , 0.6226845 ],
                  [0.53331435, 0.07999352, 0.8549948 ],
                  [0.25879037, 0.39530203, 0.698465  ],
                  [0.73427284, 0.63575995, 0.18827209],
                  [0.05689114, 0.0862954 , 0.6325046 ]], "float32")
        log_out = log_softmax(input)
        label = paddle.to_tensor([0, 2, 1, 1, 0], "int64")
        result = nll_loss(log_out, label)
        print(result) # Tensor(shape=[1], dtype=float32, place=CPUPlace, stop_gradient=True, [1.07202101])

