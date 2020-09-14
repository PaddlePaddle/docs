.. _cn_api_metric_Recall:

Recall
-------------------------------

.. py:class:: paddle.metric.Recall()


    Recall (also known as sensitivity) is the fraction of
    relevant instances that have been retrieved over the
    total amount of relevant instances

    Refer to:
    https://en.wikipedia.org/wiki/Precision_and_recall

    Noted that this class manages the recall score only for
    binary classification task.

    参数
:::::::::
        name (str, optional): String name of the metric instance.
            Default is `recall`.

    Example by standalone:
        
        .. code-block:: python

        import numpy as np
        import paddle

        x = np.array([0.1, 0.5, 0.6, 0.7])
        y = np.array([1, 0, 1, 1])

        m = paddle.metric.Recall()
        m.update(x, y)
        res = m.accumulate()
        print(res) # 2.0 / 3.0


    Example with Model API:
        
        .. code-block:: python

        import numpy as np
        
        import paddle
        import paddle.nn as nn
        
        class Data(paddle.io.Dataset):
            def __init__(self):
                super(Data, self).__init__()
                self.n = 1024
                self.x = np.random.randn(self.n, 10).astype('float32')
                self.y = np.random.randint(2, size=(self.n, 1)).astype('float32')
        
            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]
        
            def __len__(self):
                return self.n
        
        paddle.disable_static()
        model = paddle.Model(nn.Sequential(
            nn.Linear(10, 1),
            nn.Sigmoid()
        ))
        optim = paddle.optimizer.Adam(
            learning_rate=0.001, parameters=model.parameters())
        model.prepare(
            optim,
            loss=nn.BCELoss(),
            metrics=[paddle.metric.Precision(), paddle.metric.Recall()])
        
        data = Data()
        model.fit(data, batch_size=16)
    