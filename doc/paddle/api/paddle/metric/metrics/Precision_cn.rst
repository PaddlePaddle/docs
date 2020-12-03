.. _cn_api_metric_Precision:

Precision
-------------------------------

.. py:class:: paddle.metric.Precision()


精确率Precision(也称为 positive predictive value,正预测值)是被预测为正样例中实际为正的比例。 https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers 该类管理二分类任务的precision分数。

**注意**：这个metric只能用来评估二分类。


参数:
:::::::::
    - **name** (str，可选) – metric实例的名字，默认是'precision'。


**代码示例**

    # 独立使用示例:
        
        .. code-block:: python

            import numpy as np
            import paddle

            x = np.array([0.1, 0.5, 0.6, 0.7])
            y = np.array([0, 1, 1, 1])

            m = paddle.metric.Precision()
            m.update(x, y)
            res = m.accumulate()
            print(res) # 1.0

    # 在Model API中的示例:
        
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
    

.. py:function:: update(preds, labels, *args)

更新Precision的状态。

参数:
:::::::::
    - **preds** (numpy.array | Tensor): 预测输出结果通常是sigmoid函数的输出，是一个数据类型为float64或float32的向量。
    - **labels** (numpy.array | Tensor): 真实标签的shape和:code: `preds` 相同，数据类型为int32或int64。

返回: 无。


.. py:function:: reset()

清空状态和计算结果。

返回：无


.. py:function:: accumulate()

累积的统计指标，计算和返回precision值。

返回：precision值，一个标量。


.. py:function:: name()

返回Metric实例的名字, 参考上述的name，默认是'precision'。

返回: 评估的名字，string类型。
