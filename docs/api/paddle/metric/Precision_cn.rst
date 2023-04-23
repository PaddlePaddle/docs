.. _cn_api_metric_Precision:

Precision
-------------------------------

.. py:class:: paddle.metric.Precision()


精确率 Precision(也称为 positive predictive value，正预测值)是被预测为正样例中实际为正的比例。该类管理二分类任务的 precision 分数。

相关链接：https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers

.. note::
这个 metric 只能用来评估二分类。

参数
::::::::::::

    - **name** (str，可选) – metric 实例的名字，默认是'precision'。


代码示例 1
::::::::::::

独立使用示例

    .. code-block:: python

        import numpy as np
        import paddle

        x = np.array([0.1, 0.5, 0.6, 0.7])
        y = np.array([0, 1, 1, 1])

        m = paddle.metric.Precision()
        m.update(x, y)
        res = m.accumulate()
        print(res) # 1.0

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
                self.y = np.random.randint(2, size=(self.n, 1)).astype('float32')

            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]

            def __len__(self):
                return self.n

        model = paddle.Model(nn.Sequential(
            nn.Linear(10, 1),
            nn.Sigmoid()
        ))
        optim = paddle.optimizer.Adam(
            learning_rate=0.001, parameters=model.parameters())
        model.prepare(
            optim,
            loss=nn.BCELoss(),
            metrics=paddle.metric.Precision())

        data = Data()
        model.fit(data, batch_size=16)

方法
::::::::::::
update(preds, labels, *args)
'''''''''

更新 Precision 的状态。

**参数**

    - **preds** (numpy.array | Tensor)：预测输出结果通常是 sigmoid 函数的输出，是一个数据类型为 float64 或 float32 的向量。
    - **labels** (numpy.array | Tensor)：真实标签的 shape 和：code: `preds` 相同，数据类型为 int32 或 int64。

**返回**

 无。

reset()
'''''''''

清空状态和计算结果。

**返回**

无。


accumulate()
'''''''''

累积的统计指标，计算和返回 precision 值。

**返回**

precision 值，一个标量。


name()
'''''''''

返回 Metric 实例的名字，参考上述的 name，默认是'precision'。

**返回**

评估的名字，string 类型。
