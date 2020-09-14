.. _cn_api_nn_cn_margin_ranking_loss:

margin_ranking_loss
-------------------------------

.. py:function:: paddle.nn.functional.margin_ranking_loss(input, other, label, margin=0.0, reduction='mean', name=None)

该算子计算输入input，other 和 标签label间的 `margin rank loss` 损失。该损失函数的数学计算公式如下：

 .. math:: 
     margin\_rank\_loss = max(0, -label * (input - other) + margin)

当 `reduction` 设置为 ``'mean'`` 时，

    .. math::
       Out = MEAN(margin\_rank\_loss)

当 `reduction` 设置为 ``'sum'`` 时，
    
    .. math::
       Out = SUM(margin\_rank\_loss)

当 `reduction` 设置为 ``'none'`` 时，直接返回最原始的 `margin_rank_loss` 。

参数
::::::::
    - **input** （Tensor）：第一个输入的 `Tensor` ，数据类型为：float32、float64。
    - **other** （Tensor）：第二个输入的 `Tensor` ，数据类型为：float32、float64。
    - **label** （Tensor）：训练数据的标签，数据类型为：float32, float64。
    - **margin** （float，可选）： - 用于加和的margin值，默认值为0。  
    - **reduction** （string，可选）： - 指定应用于输出结果的计算方式，可选值有: ``'none'`` 、 ``'mean'`` 、 ``'sum'`` 。如果设置为 ``'none'`` ，则直接返回 最原始的 ``margin_rank_loss`` 。如果设置为 ``'sum'`` ，则返回 ``margin_rank_loss`` 的总和。如果设置为 ``'mean'`` ，则返回 ``margin_rank_loss`` 的平均值。默认值为 ``'none'`` 。
    - **name** （str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。  

返回
::::::::
Tensor, 如果 :attr:`reduction` 为 ``'sum'`` 或者是 ``'mean'`` ，则形状为 :math:`[1]` ，否则shape和输入 `input` 保持一致 。数据类型与 ``input``、 ``other`` 相同。

代码示例
::::::::

.. code-block:: python

    import paddle
    paddle.disable_static()
     
    input = paddle.to_tensor([[1, 2], [3, 4]], dtype='float32')
    other = paddle.to_tensor([[2, 1], [2, 4]], dtype='float32')
    label = paddle.to_tensor([[1, -1], [-1, -1]], dtype='float32')
    loss = paddle.nn.functional.margin_ranking_loss(input, other, label) 
    print(loss.numpy()) # [0.75]
