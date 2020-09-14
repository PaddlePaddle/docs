.. _cn_api_metric_Accuracy:

Accuracy
-------------------------------

.. py:class:: paddle.metric.Accuracy()

计算准确率(accuracy)。

参数：
:::::::::
    - **topk** (int|tuple(int)) - 计算准确率的top个数，默认是1。
    - **name** (str, optional) - metric实例的名字，默认是'acc'。

**代码示例**：

    # 独立使用示例:
        
        .. code-block:: python

            import numpy as np
            import paddle

            paddle.disable_static()
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


    # 在Model API中的示例:
        
        .. code-block:: python

            import paddle

            paddle.disable_static()
            train_dataset = paddle.vision.datasets.MNIST(mode='train')

            model = paddle.Model(paddle.vision.LeNet(classifier_activation=None))
            optim = paddle.optimizer.Adam(
                learning_rate=0.001, parameters=model.parameters())
            model.prepare(
                optim,
                loss=paddle.nn.CrossEntropyLoss(),
                metrics=paddle.metric.Accuracy())

            model.fit(train_dataset, batch_size=64)


.. py:function:: compute(pred, label, *args)

计算top-k（topk中的最大值）的索引。

参数：
:::::::::
    - **pred**  (Tensor) - 预测结果为是float64或float32类型的Tensor。
    - **label**  (Tensor) - 真实的标签值是一个2D的Tensor，shape为[batch_size, 1], 数据类型为int64。

返回: 一个Tensor，shape是[batch_size, topk], 值为0或1，1表示预测正确.


.. py:function:: update(pred, label, *args)

更新metric的状态（正确预测的个数和总个数），以便计算累积的准确率。返回当前step的准确率。

参数:
:::::::::
    - **correct** (numpy.array | Tensor): 一个值为0或1的Tensor，shape是[batch_size, topk]。

返回: 当前step的准确率。


.. py:function:: reset()

清空状态和计算结果。

返回
:::::::::
  无


.. py:function:: accumulate()

累积的统计指标，计算和返回准确率。

返回
:::::::::
  准确率，一般是个标量 或 多个标量，和topk的个数一致。


.. py:function:: name()

返回Metric实例的名字, 参考上述name，默认是'acc'。

返回
:::::::::
  评估的名字，string类型。
