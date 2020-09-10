.. _cn_api_metric_Accuracy:

Accuracy
-------------------------------

.. py:class:: paddle.metric.Accuracy()


    Encapsulates accuracy metric logic.

    参数
:::::::::
        topk (int|tuple(int)): Number of top elements to look at
            for computing accuracy. Default is (1,).
        name (str, optional): String name of the metric instance. Default
            is `acc`.

    Example by standalone:
        
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


    Example with Model API:
        
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

    