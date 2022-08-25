.. _cn_api_metric_Accuracy:

Accuracy
-------------------------------

.. py:class:: paddle.metric.Accuracy(topk=(1, ), name=None, *args, **kwargs)


计算准确率(accuracy)。

参数：
:::::::::
    - **topk** (list[int]|tuple[int]，可选) - 计算准确率的 top 个数，默认值为 (1,)。
    - **name** (str，可选) - metric 实例的名字。默认值为 None，表示使用默认名字 'acc'。

代码示例
:::::::::

**独立使用示例:**

    .. code-block:: python

        import numpy as np
        import paddle
        x = paddle.to_tensor(np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.4, 0.3, 0.2],
            [0.1, 0.2, 0.4, 0.3],
            [0.1, 0.2, 0.3, 0.4]]))
        y = paddle.to_tensor(np.array([[0], [1], [2], [3]]))
        m = paddle.metric.Accuracy()
        correct = m.compute(x, y)
        m.update(correct)
        res = m.accumulate()
        print(res) # 0.75


**在 Model API 中的示例**

    .. code-block:: python

        import paddle
        from paddle.static import InputSpec
        import paddle.vision.transforms as T
        from paddle.vision.datasets import MNIST

        input = InputSpec([None, 1, 28, 28], 'float32', 'image')
        label = InputSpec([None, 1], 'int64', 'label')
        transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
        train_dataset = MNIST(mode='train', transform=transform)

        model = paddle.Model(paddle.vision.models.LeNet(), input, label)
        optim = paddle.optimizer.Adam(
            learning_rate=0.001, parameters=model.parameters())
        model.prepare(
            optim,
            loss=paddle.nn.CrossEntropyLoss(),
            metrics=paddle.metric.Accuracy())

        model.fit(train_dataset, batch_size=64)


compute(pred, label, *args)
:::::::::

计算 top-k（topk 中的最大值）的索引。

**参数**

    - **pred** (Tensor) - 预测结果为是 float64 或 float32 类型的 Tensor。shape 为[batch_size, d0, ..., dN].
    - **label** (Tensor) - 真实的标签值是一个 int64 类型的 Tensor，shape 为[batch_size, d0, ..., 1] 或 one hot 表示的形状[batch_size, d0, ..., num_classes].

**返回**

Tensor，shape 是[batch_size, d0, ..., topk], 值为 0 或 1，1 表示预测正确.


update(correct, *args)
:::::::::

更新 metric 的状态（正确预测的个数和总个数），以便计算累积的准确率。返回当前 step 的准确率。

**参数**

    - **correct** (numpy.array | Tensor): 一个值为 0 或 1 的 Tensor，shape 是[batch_size, d0, ..., topk]。

**返回**

当前 step 的准确率。

reset()
:::::::::

清空状态和计算结果。

accumulate()
:::::::::

累积的统计指标，计算和返回准确率。

**返回**

准确率，一般是个标量 或 多个标量，和 topk 的个数一致。


name()
:::::::::

返回 Metric 实例的名字, 参考上述 name，默认是'acc'。

**返回**

评估的名字，string 类型。
