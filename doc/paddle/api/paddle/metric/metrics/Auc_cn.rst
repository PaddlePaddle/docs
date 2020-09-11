.. _cn_api_metric_Auc:

Auc
-------------------------------

.. py:class:: paddle.metric.Auc()


    The auc metric is for binary classification.
    Refer to https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve.
    Please notice that the auc metric is implemented with python, which may be a little bit slow.

    The `auc` function creates four local variables, `true_positives`,
    `true_negatives`, `false_positives` and `false_negatives` that are used to
    compute the AUC. To discretize the AUC curve, a linearly spaced set of
    thresholds is used to compute pairs of recall and precision values. The area
    under the ROC-curve is therefore computed using the height of the recall
    values by the false positive rate, while the area under the PR-curve is the
    computed using the height of the precision values by the recall.

    参数
:::::::::
        curve (str): Specifies the mode of the curve to be computed,
            'ROC' or 'PR' for the Precision-Recall-curve. Default is 'ROC'.
        num_thresholds (int): The number of thresholds to use when
            discretizing the roc curve. Default is 4095.
            'ROC' or 'PR' for the Precision-Recall-curve. Default is 'ROC'.
        name (str, optional): String name of the metric instance. Default
            is `auc`.

    "NOTE: only implement the ROC curve type via Python now."

    Example by standalone:
        .. code-block:: python

        import numpy as np
        import paddle

        m = paddle.metric.Auc()
        
        n = 8
        class0_preds = np.random.random(size = (n, 1))
        class1_preds = 1 - class0_preds
        
        preds = np.concatenate((class0_preds, class1_preds), axis=1)
        labels = np.random.randint(2, size = (n, 1))
        
        m.update(preds=preds, labels=labels)
        res = m.accumulate()


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
                self.y = np.random.randint(2, size=(self.n, 1)).astype('int64')
        
            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]
        
            def __len__(self):
                return self.n
        
        paddle.disable_static()
        model = paddle.Model(nn.Sequential(
            nn.Linear(10, 2), nn.Softmax())
        )
        optim = paddle.optimizer.Adam(
            learning_rate=0.001, parameters=model.parameters())
        
        def loss(x, y):
            return nn.functional.nll_loss(paddle.log(x), y)
        
        model.prepare(
            optim,
            loss=loss,
            metrics=paddle.metric.Auc())
        data = Data()
        model.fit(data, batch_size=16)
    