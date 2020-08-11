.. _cn_api_nn_loss_MarginRankingLoss:
MarginRankingLoss
-------------------------------

.. py:class:: paddle.nn.loss.MarginRankingLoss(margin=0.0, reduction='mean', name=None)

该接口用于创建一个 ``MarginRankingLoss`` 的可调用类，计算输入x1，x2 和 标签target间的 `margin rank loss` 损失。

该损失函数的数学计算公式如下：

 .. math:: 
     margin\_rank\_loss = max(0, -target * (x1 - x2) + margin)

当 `reduction` 设置为 ``'mean'`` 时，

    .. math::
       Out = MEAN(margin\_rank\_loss)

当 `reduction` 设置为 ``'sum'`` 时，
    
    .. math::
       Out = SUM(margin\_rank\_loss)

当 `reduction` 设置为 ``'none'`` 时，直接返回最原始的 `margin_rank_loss` 。

参数
::::::::
    - **margin** （float，可选）： - 用于加和的margin值，默认值为0。  
    - **reduction** （string，可选）： - 指定应用于输出结果的计算方式，可选值有: ``'none'`` 、 ``'mean'`` 、 ``'sum'`` 。如果设置为 ``'none'`` ，则直接返回 最原始的 ``margin_rank_loss`` 。如果设置为 ``'sum'`` ，则返回 ``margin_rank_loss`` 的总和。如果设置为 ``'mean'`` ，则返回 ``margin_rank_loss`` 的平均值。默认值为 ``'none'`` 。
    - **name** （str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

形状
::::::::
    - **x1** - N-D Tensor, 维度是[N，*] 其中N 是batch size，`*` 是任意数量的额外维度，数据类型为float32或float64。
    - **x2** - 与 ``x1`` 的形状、数据类型相同。
    - **label** - 与 ``x1`` 的形状、数据类型相同。
    - **out** - 如果 :attr:`reduction` 为 ``'sum'`` 或者是 ``'mean'`` ，则形状为 :math:`[1]` ，否则shape和输入 `x1` 保持一致 。数据类型与 ``x1``、 ``x2`` 相同。

返回
::::::::
返回计算MarginRankingLoss的可调用对象。

代码示例
::::::::

.. code-block:: python


     import numpy as np 
     import paddle 
     
     paddle.disable_static()
      
     x1 = paddle.to_variable(np.array([[1, 2], [3, 4]]).astype("float32"))
     x2 = paddle.to_variable(np.array([[2, 1], [2, 4]]).astype("float32"))
     target = paddle.to_variable(np.array([[1, -1], [-1, -1]]).astype("float32"))
     margin_rank_loss = paddle.nn.MarginRankingLoss()
     loss = margin_rank_loss(x1, x2, target) 
     print(loss.numpy()) # [0.75]
