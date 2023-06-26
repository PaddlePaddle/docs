.. _cn_api_metric_Auc:

Auc
-------------------------------

.. py:class:: paddle.metric.Auc()

.. note::
目前只用 Python 实现 Auc，可能速度略慢。

该接口计算 Auc，在二分类(binary classification)中广泛使用。

该接口创建四个局部变量 true_positives，true_negatives，false_positives 和 false_negatives，用于计算 Auc。为了离散化 AUC 曲线，使用临界值的线性间隔来计算召回率和准确率的值。用 false positive 的召回值高度计算 ROC 曲线面积，用 recall 的准确值高度计算 PR 曲线面积。

参考链接：https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve

参数
::::::::::::

    - **curve** (str) - 将要计算的曲线名的模式，包括'ROC'（默认）或者'PR'（Precision-Recall-curve）。
    - **num_thresholds** (int) - 离散化 AUC 曲线的整数阈值数，默认是 4095。
    - **name** (str，可选) - metric 实例的名字，默认是'auc'。

代码示例 1
::::::::::::

独立使用示例

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

代码示例 2
::::::::::::

在 Model API 中的示例

    .. code-block:: python

        import numpy as np
        import paddle
        import paddle.nn as nn

        class Data(paddle.io.Dataset):
            def __init__(self):
                super().__init__()
                self.n = 1024
                self.x = np.random.randn(self.n, 10).astype('float32')
                self.y = np.random.randint(2, size=(self.n, 1)).astype('int64')

            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]

            def __len__(self):
                return self.n

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


方法
::::::::::::
update(pred, label, *args)
'''''''''

更新 AUC 计算的状态。

**参数**

    - **preds** (numpy.array | Tensor)：一个 shape 为[batch_size, 2]的 Numpy 数组或 Tensor，preds[i][j]表示第 i 个样本类别为 j 的概率。
    - **labels** (numpy.array | Tensor)：一个 shape 为[batch_size, 1]的 Numpy 数组或 Tensor，labels[i]是 0 或 1，表示第 i 个样本的类别。

**返回**

无。


reset()
'''''''''

清空状态和计算结果。

**返回**

无。


accumulate()
'''''''''

累积的统计指标，计算和返回 AUC 值。

**返回**

AUC 值，一个标量。


name()
'''''''''

返回 Metric 实例的名字，参考上述的 name，默认是'auc'。

**返回**

 评估的名字，string 类型。
