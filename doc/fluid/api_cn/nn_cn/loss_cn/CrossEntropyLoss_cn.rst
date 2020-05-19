CrossEntropyLoss
-------------------------------

.. py:function:: paddle.nn.loss.CrossEntropyLoss(weight=None, reduction='mean', ignore_index=-100)

该OP计算输入input和标签label间的交叉熵损失 ，它结合了`LogSoftmax` 和 `NLLLoss` 的OP计算，可用于训练一个 `n` 类分类器。

如果提供 `weight` 参数的话，它是一个 `1-D` 的tensor, 每个值对应每个类别的权重。
该损失函数的数学计算公式如下：

    .. math::
        loss_j =  -\text{input[class]} +
        \log\left(\sum_{i=0}^{K}\exp(\text{input}_i)\right), j = 1,..., K

当 `weight` 不为 `none` 时，损失函数的数学计算公式为：

    .. math::
        loss_j =  \text{weight[class]}(-\text{input[class]} +
        \log\left(\sum_{i=0}^{K}\exp(\text{input}_i)\right)), j = 1,..., K


参数：
    - **input** (Variable): - 输入 `Tensor`，数据类型为float32或float64。其形状为 :math:`[N, C]` , 其中 `C` 为类别数。对于多维度的情形下，它的形状为 :math:`[N, C, d_1, d_2, ..., d_k]`，k >= 1。
    - **label** (Variable): - 输入input对应的标签值，数据类型为int64。其形状为 :math:`[N]`，每个元素符合条件：0 <= label[i] <= C-1。对于多维度的情形下，它的形状为 :math:`[N, d_1, d_2, ..., d_k]`，k >= 1。
    - **weight** (Variable, 可选): - 指定每个类别的权重。其默认为 `None` 。如果提供该参数的话，维度必须为 `C`（类别数）。数据类型为float32或float64。
    - **reduction** (string, 可选): - 指定应用于输出结果的计算方式，数据类型为string，可选值有: `none`, `mean`, `sum` 。默认为 `mean` ，计算 `mini-batch` loss均值。设置为 `sum` 时，计算 `mini-batch` loss的总和。设置为 `none` 时，则返回loss Tensor。
    - **ignore_index** (int64, 可选): - 指定一个忽略的标签值，此标签值不参与计算。默认值为-100。数据类型为int64。

返回：返回计算 `CrossEntropyLoss` 交叉熵后的损失值。

返回类型：Variable

**代码示例**

..  code-block:: python

            # declarative mode
            import paddle
            import paddle.fluid as fluid
            import numpy as np
            input = fluid.data(name='input', shape=[5, 100], dtype='float64')
            label = fluid.data(name='label', shape=[5], dtype='int64')
            weight = fluid.data(name='weight', shape=[100], dtype='float64')
            ce_loss = paddle.nn.loss.CrossEntropyLoss(weight=weight, reduction='mean')
            output = ce_loss(input, label)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            input_data = np.random.random([5, 100]).astype("float64")
            label_data = np.random.randint(0, 100, size=(5)).astype(np.int64)
            weight_data = np.random.random([100]).astype("float64")
            output = exe.run(fluid.default_main_program(),
                        feed={"input": input_data, "label": label_data,"weight": weight_data},
                        fetch_list=[output],
                        return_numpy=True)
            print(output)
            # imperative mode
            import paddle.fluid.dygraph as dg
            with dg.guard(place) as g:
                input = dg.to_variable(input_data)
                label = dg.to_variable(label_data)
                weight = dg.to_variable(weight_data)
                ce_loss = paddle.nn.loss.CrossEntropyLoss(weight=weight, reduction='mean')
                output = ce_loss(input, label)
                print(output.numpy())

