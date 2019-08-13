.. _cn_api_fluid_layers_center_loss:

center_loss
-------------------------------

.. py:function:: paddle.fluid.layers.center_loss(input, label, num_classes, alpha, param_attr, update_center=True)

center_loss层
该层接收一个来自于最后一个隐藏层的输出和目标标签作为输入，返回损失值。
对于输入，\(X\)和标签\(Y\)，计算公式为：

    .. math::

        out = \frac{1}{2}(X - Y)^2



参数:

    - **input** (Variable) - 形为[N x M]的2维张量
    - **label** (Variable) - groud truth,一个形为[N x 1]的2维张量，N表示batch size
    - **num_class** (int) - 表示类别的数
    - **alpha** (float|Variable) - 学习率
    - **param_attr** (ParamAttr) - 参数初始化
    - **update_center** (bool) - 是否更新损失值

返回：形为[N x 1]的2维张量。

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









