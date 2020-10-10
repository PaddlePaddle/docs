.. _cn_api_distribution_Categorical:

Categorical
-------------------------------

.. py:class:: paddle.distribution.Categorical(logits, name=None)




类别分布是一种离散概率分布，其随机变量可以取K个相互独立类别的其中一个。

概率质量函数（pmf）为：

.. math::

    pmf(k; p_i) =\prod_{i=1}^{k} p_i^{[x=i]}

上面公式中:
  - :math:`[x = i]` 表示：如果 :math:`x==i` ，则表达式取值为1，否则取值为0。


参数：
    - **logits** (list|numpy.ndarray|Tensor) - 类别分布对应的logits。数据类型为float32或float64。
    - **name** (str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

**代码示例**：

.. code-block:: python

    import paddle
    from paddle.distribution import Categorical

    x = paddle.rand([6])
    print(x.numpy())
    # [0.32564053, 0.99334985, 0.99034804,
    #  0.09053693, 0.30820143, 0.19095989]
    y = paddle.rand([6])
    print(y.numpy())
    # [0.6365463 , 0.7278677 , 0.90260243, 
    # 0.5226815 , 0.35837543, 0.13981032]

    cat = Categorical(x)
    cat2 = Categorical(y)

    cat.sample([2,3])
    # [[5, 1, 1],
    # [0, 1, 2]]

    cat.entropy()
    # [1.71887]

    cat.kl_divergence(cat2)
    # [0.0278455]

    value = paddle.to_tensor([2,1,3])
    cat.probs(value)
    # [0.341613 0.342648 0.03123]

    cat.log_prob(value)
    # [-1.07408 -1.07105 -3.46638]


.. py:function:: sample(shape)

生成指定维度的样本

参数：
    - **shape** (list) - 指定生成样本的维度。

返回：预先设计好维度的张量

返回类型：Tensor

代码示例：

.. code-block:: python

    import paddle
    from paddle.distribution import Categorical

    x = paddle.rand([6])
    print(x.numpy())
    # [0.32564053, 0.99334985, 0.99034804,
    #  0.09053693, 0.30820143, 0.19095989]

    cat = Categorical(x)

    cat.sample([2,3])
    # [[5, 1, 1],
    # [0, 1, 2]]

.. py:function:: kl_divergence(other)

相对于另一个类别分布的KL散度

参数：
    - **other** (Categorical) - 输入的另一个类别分布。数据类型为float32。
    
返回：相对于另一个类别分布的KL散度, 数据类型为float32

返回类型：Tensor

代码示例：

.. code-block:: python

    import paddle
    from paddle.distribution import Categorical

    x = paddle.rand([6])
    print(x.numpy())
    # [0.32564053, 0.99334985, 0.99034804,
    #  0.09053693, 0.30820143, 0.19095989]
    y = paddle.rand([6])
    print(y.numpy())
    # [0.6365463 , 0.7278677 , 0.90260243, 
    # 0.5226815 , 0.35837543, 0.13981032]

    cat = Categorical(x)
    cat2 = Categorical(y)

    cat.kl_divergence(cat2)
    # [0.0278455]

.. py:function:: entropy()

信息熵
    
返回：类别分布的信息熵, 数据类型为float32

返回类型：Tensor

代码示例：

.. code-block:: python

    import paddle
    from paddle.distribution import Categorical

    x = paddle.rand([6])
    print(x.numpy())
    # [0.32564053, 0.99334985, 0.99034804,
    #  0.09053693, 0.30820143, 0.19095989]

    cat = Categorical(x)

    cat.entropy()
    # [1.71887]

.. py:function:: probs(value)

所选择类别的概率。
如果 ``logtis`` 是2-D或更高阶的Tensor，那么其最后一个维度表示不同类别的概率，其它维度被看做不同的概率分布。
同时，如果 ``value`` 是1-D Tensor，那么 ``value`` 会broadcast成与 ``logits`` 具有相同的概率分布数量。
如果 ``value`` 为更高阶Tensor，那么 ``value`` 应该与 ``logits`` 具有相同的概率分布数量。也就是说， ``value[:-1] = logits[:-1]`` 。

参数：
    - **value** (Tensor) - 输入张量, 表示选择的类别下标。数据类型为int32或int64。

返回：给定类别下标的概率

返回类型：Tensor

.. code-block:: python

    import paddle
    from paddle.distribution import Categorical

    x = paddle.rand([6])
    print(x.numpy())
    # [0.32564053, 0.99334985, 0.99034804,
    #  0.09053693, 0.30820143, 0.19095989]

    cat = Categorical(x)

    value = paddle.to_tensor([2,1,3])
    cat.probs(value)
    # [0.341613 0.342648 0.03123]

.. py:function:: log_prob(value)

所选择类别的对数概率

参数：
    - **value** (Tensor) - 输入张量, 表示选择的类别下标。数据类型为int32或int64。

返回：对数概率

返回类型：Tensor

.. code-block:: python

    import paddle
    from paddle.distribution import Categorical

    x = paddle.rand([6])
    print(x.numpy())
    # [0.32564053, 0.99334985, 0.99034804,
    #  0.09053693, 0.30820143, 0.19095989]

    cat = Categorical(x)

    value = paddle.to_tensor([2,1,3])

    cat.log_prob(value)
    # [-1.07408 -1.07105 -3.46638]





