.. _cn_api_nn_cn_MarginRankingLoss:

MarginRankingLoss

-------------------------------

.. py:function:: paddle.nn.loss.MarginRankingLoss(margin=0.0, reduction='mean')

该接口用于创建一个 ``MarginRankingLoss`` 的可调用类，计算输入x, y 和 标签label间的 `margin rank loss` 损失。

该损失函数的数学计算公式如下：
 .. math:: 
     margin_rank_loss = max(0, -label * (x- y) + margin)

当 `reduction` 设置为 ``'mean'`` 时，

    .. math::
       Out = MEAN(margi_rank_loss)

当 `reduction` 设置为 ``'sum'`` 时，
    
    .. math::
       Out = SUM(margin_rank_loss)

当 `reduction` 设置为 ``'none'`` 时，直接返回最原始的 `margin_rank_loss` 。

参数：
    - **margin** (float，可选)： - 用于加和的margin值，默认值为0。  
    - **reduction** (string, 可选)： - 指定应用于输出结果的计算方式，可选值有: ``'none'`` | ``'mean'`` |  ``'sum'`` 。
            如果设置为 ``'none'`` ，则直接返回 ``margin_rank_loss`` 。
            如果设置为 ``'sum'`` ，则返回 ``margin_rank_loss`` 的总和。
            如果设置为 ``'mean'`` ，则返回 ``margin_rank_loss`` 的平均值。
            默认值为 ``'none'`` 。
返回：返回计算MarginRankingLoss的可调用对象。

**代码示例**

.. code-block:: python
     import numpy as np 
     import paddle 
     import paddle.imperative as imperative
     
     paddle.enable_imperative()
      
     x = imperative.to_variable(np.array([[1, 2], [3, 4]]))
     y = imperative.to_variable(np.array([[2, 1], [2, 4]]))
     label = imperative.to_variable(np.array([[1, -1], [-1, -1]]))
     margin_rank_loss = MarginRankingLoss()
     loss = margin_rank_loss(x, y, label) 
     print(loss.numpy()) # [0.75]
