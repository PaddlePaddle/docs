.. _cn_api_fluid_layers_center_loss:

center_loss
-------------------------------

.. py:function:: paddle.fluid.layers.center_loss(input, label, num_classes, alpha, param_attr, update_center=True)

center_loss层
该OP接收一个来自于最后一个隐藏层的输出和目标标签作为输入，返回损失值。为每一个类别提供一个类别中心，计算mini-batch中每个样本与对应类别中心的距离的平均值作为center loss。

对于输入，\(X\)和标签\(Y\)，计算公式为：

    .. math::

        out = \frac{1}{2}(X - Y)^2



参数:

    - **input** (Variable) - 输入形状为[N x M]的2维张量，数据类型为float32，float64。
    - **label** (Variable) - 输入的标签，一个形状为为[N x 1]的2维张量，N表示batch size，数据类型为int32。
    - **num_class** (int32) - 输入类别的数量。
    - **alpha** (float32|float64|Variable) - 学习率。数据类型为float32或者float64。
    - **param_attr** (ParamAttr) - 指定权重参数属性的对象。表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **update_center** (bool) - 是否更新类别中心的参数。

返回：形状为[N x 1]的2维Tensor|LoDTensor。

返回类型：Variable

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        
        input = fluid.layers.data(name='x',shape=[20,30],dtype='float32')
        label = fluid.layers.data(name='y',shape=[20,1],dtype='int64')
        num_classes = 1000
        alpha = 0.01
        param_attr = fluid.initializer.Xavier(uniform=False)
        center_loss=fluid.layers.center_loss(input=input,
            label=label,
            num_classes=1000,
            alpha=alpha,
            param_attr=fluid.initializer.Xavier(uniform=False),
            update_center=True)









