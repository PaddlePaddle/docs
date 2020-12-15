.. _cn_api_metric_Auc:

Auc
-------------------------------

.. py:class:: paddle.metric.Auc()

**注意**：目前只用Python实现Auc，可能速度略慢

该接口计算Auc，在二分类(binary classification)中广泛使用。相关定义参考 https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve 。

该接口创建四个局部变量true_positives, true_negatives, false_positives和false_negatives，用于计算Auc。为了离散化AUC曲线，使用临界值的线性间隔来计算召回率和准确率的值。用false positive的召回值高度计算ROC曲线面积，用recall的准确值高度计算PR曲线面积。


参数：
:::::::::
    - **curve** (str) - 将要计算的曲线名的模式，包括'ROC'（默认）或者'PR'（Precision-Recall-curve）。
    - **num_thresholds** (int) - 离散化AUC曲线的整数阈值数，默认是4095。
    - **name** (str，可选) – metric实例的名字，默认是'auc'。

**代码示例**：

    # 独立使用示例:

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
    

.. py:function:: update(pred, label, *args)

更新AUC计算的状态。

参数:
:::::::::
    - **preds** (numpy.array | Tensor): 一个shape为[batch_size, 2]的Numpy数组或Tensor，preds[i][j]表示第i个样本类别为j的概率。
    - **labels** (numpy.array | Tensor): 一个shape为[batch_size, 1]的Numpy数组或Tensor，labels[i]是0或1，表示第i个样本的类别。

返回: 无。


.. py:function:: reset()

清空状态和计算结果。

返回：无


.. py:function:: accumulate()

累积的统计指标，计算和返回AUC值。

返回：AUC值，一个标量。


.. py:function:: name()

返回Metric实例的名字, 参考上述的name，默认是'auc'。

返回: 评估的名字，string类型。
