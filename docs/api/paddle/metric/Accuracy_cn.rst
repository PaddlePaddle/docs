.. _cn_api_metric_Accuracy:

Accuracy
-------------------------------

.. py:class:: paddle.metric.Accuracy()

计算准确率(accuracy)。

参数：
:::::::::
    - **topk** (int|tuple(int)) - 计算准确率的top个数，默认是1。
    - **name** (str, optional) - metric实例的名字，默认是'acc'。

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


**在Model API中的示例**
        
    .. code-block:: python

        import paddle
        from paddle.static import InputSpec
        import paddle.vision.transforms as T
        from paddle.vision.datasets import MNIST
               
        input = InputSpec([None, 1, 28, 28], 'float32', 'image')
        label = InputSpec([None, 1], 'int64', 'label')
        transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
        train_dataset = MNIST(mode='train', transform=transform)
  
        model = paddle.Model(paddle.vision.LeNet(), input, label)
        optim = paddle.optimizer.Adam(
            learning_rate=0.001, parameters=model.parameters())
        model.prepare(
            optim,
            loss=paddle.nn.CrossEntropyLoss(),
            metrics=paddle.metric.Accuracy())
  
        model.fit(train_dataset, batch_size=64)



compute(pred, label, *args)
:::::::::

计算top-k（topk中的最大值）的索引。

**参数**
    
    - **pred**  (Tensor) - 预测结果为是float64或float32类型的Tensor。shape为[batch_size, d0, ..., dN].
    - **label**  (Tensor) - 真实的标签值是一个int64类型的Tensor，shape为[batch_size, d0, ..., 1] 或one hot表示的形状[batch_size, d0, ..., num_classes].

**返回**: Tensor，shape是[batch_size, d0, ..., topk], 值为0或1，1表示预测正确.


update(pred, label, *args)
:::::::::

更新metric的状态（正确预测的个数和总个数），以便计算累积的准确率。返回当前step的准确率。

**参数:**

    - **correct** (numpy.array | Tensor): 一个值为0或1的Tensor，shape是[batch_size, d0, ..., topk]。

**返回:** 当前step的准确率。


reset()
:::::::::

清空状态和计算结果。

accumulate()
:::::::::

累积的统计指标，计算和返回准确率。

**返回:** 准确率，一般是个标量 或 多个标量，和topk的个数一致。


name()
:::::::::

返回Metric实例的名字, 参考上述name，默认是'acc'。

**返回:** 评估的名字，string类型。
