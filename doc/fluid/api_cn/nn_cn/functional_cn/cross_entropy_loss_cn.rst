.. _cn_api_nn_functional_cross_entropy_loss:

cross_entropy_loss
-------------------------------
.. py:function:: paddle.nn.functional.cross_entropy_loss(input, label, weight=None, ignore_index=-100, reduction='mean')

该接口计算输入input和标签label间的交叉熵损失 ，它结合了 `LogSoftmax` 和 `NLLLoss` 的计算，可用于训练一个 `n` 类分类器。

如果提供 `weight` 参数的话，它是一个 `1-D` 的tensor, 每个值对应每个类别的权重。
该损失函数的数学计算公式如下：

    .. math::
        loss_j =  -\text{input[class]} +
        \log\left(\sum_{i=0}^{K}\exp(\text{input}_i)\right), j = 1,..., K

当 `weight` 不为 `none` 时，损失函数的数学计算公式为：

    .. math::
        loss_j =  \text{weight[class]}(-\text{input[class]} +
        \log\left(\sum_{i=0}^{K}\exp(\text{input}_i)\right)), j = 1,..., K


参数
:::::::::
    - **input** (Tensor): - 输入 `Tensor` ，数据类型为float32或float64。其形状为 :math:`[N, C]` , 其中 `C` 为类别数。对于多维度的情形下，它的形状为 :math:`[N, C, d_1, d_2, ..., d_k]` ，k >= 1。
    - **label** (Tensor): - 输入input对应的标签值，数据类型为int64。其形状为 :math:`[N]` ，每个元素符合条件：0 <= label[i] <= C-1。对于多维度的情形下，它的形状为 :math:`[N, d_1, d_2, ..., d_k]` ，k >= 1。
    - **weight** (Tensor, 可选): - 指定每个类别的权重。其默认为 `None` 。如果提供该参数的话，维度必须为 `C` （类别数）。数据类型为float32或float64。
    - **ignore_index** (int64, 可选): - 指定一个忽略的标签值，此标签值不参与计算。默认值为-100。数据类型为int64。
    - **reduction** (str, 可选): - 指定应用于输出结果的计算方式，数据类型为string，可选值有: `none` , `mean` , `sum` 。默认为 `mean` ，计算 `mini-batch` loss均值。设置为 `sum` 时，计算 `mini-batch` loss的总和。设置为 `none` 时，则返回loss Tensor。

返回
:::::::::
 `Tensor` , 返回计算 `cross_entropy_loss` 交叉熵后的损失值。


代码示例
:::::::::

..  code-block:: python

            import paddle
            paddle.disable_static()
            input_data = np.random.random([5, 100]).astype("float64")
            label_data = np.random.randint(0, 100, size=(5)).astype(np.int64)
            weight_data = np.random.random([100]).astype("float64")
            input =  paddle.to_tensor(input_data)
            label =  paddle.to_tensor(label_data)
            weight = paddle.to_tensor(weight_data)
            loss = paddle.nn.functional.cross_entropy(input=input, label=label, weight=weight)
            print(loss.numpy())

