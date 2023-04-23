.. _cn_api_metric_Metric:

Metric
-------------------------------

.. py:class:: paddle.metric.Metric()


评估器 metric 的基类。

用法：

    .. code-block:: text

        m = SomeMetric()
        for prediction, label in ...:
            m.update(prediction, label)
        m.accumulate()

`compute` 接口的进阶用法：

在 `compute` 中可以使用 PaddlePaddle 内置的算子进行评估器的状态，而不是通过
Python/NumPy，这样可以加速计算。`update` 接口将 `compute` 的输出作为
输入，内部采用 Python/NumPy 计算。

`Metric` 计算流程如下 （在{}中的表示模型和评估器的计算）:

    .. code-block:: text

             inputs & labels              || ------------------
                   |                      ||
                {model}                   ||
                   |                      ||
            outputs & labels              ||
                   |                      ||    tensor data
            {Metric.compute}              ||
                   |                      ||
          metric states(tensor)           ||
                   |                      ||
            {fetch as numpy}              || ------------------
                   |                      ||
          metric states(numpy)            ||    numpy data
                   |                      ||
            {Metric.update}               \/ ------------------

代码示例 1
::::::::::::

以 计算正确率的 `Accuracy` 为例，该评估器的输入为 `pred` 和 `label`，可以在 `compute` 中通过 `pred` 和 `label`先计算正确预测的矩阵。
例如，预测结果包含 10 类，`pred` 的 shape 是[N, 10]，`label` 的 shape 是[N, 1]，N 是 batch size，我们需要计算 top-1 和 top-5 的准确率，
可以在 `compute` 中计算每个样本的 top-5 得分，正确预测的矩阵的 shape 是[N, 5]。


    .. code-block:: python

        def compute(pred, label):
            # sort prediction and slice the top-5 scores
            pred = paddle.argsort(pred, descending=True)[:, :5]
            # calculate whether the predictions are correct
            correct = pred == label
            return paddle.cast(correct, dtype='float32')

代码示例 2
::::::::::::

在 `compute` 中的计算，使用内置的算子(可以跑在 GPU 上，使得速度更快)。作为 `update` 的输入，该接口计算如下：

    .. code-block:: python

        def update(self, correct):
            accs = []
            for i, k in enumerate(self.topk):
                num_corrects = correct[:, :k].sum()
                num_samples = len(correct)
                accs.append(float(num_corrects) / num_samples)
                self.total[i] += num_corrects
                self.count[i] += num_samples
            return accs

方法
::::::::::::
reset()
'''''''''

清空状态和计算结果。

**返回**

无。


update(*args)
'''''''''

更新状态。如果定义了 `compute` ， `update` 的输入是 `compute` 的输出。如果没有定义，则输入是网络的输出**output**和标签**label**，
如：`update(output1, output2, ..., label1, label2,...)` 。

也可以参考 `update` 。


accumulate()
'''''''''

累积的统计指标，计算和返回评估结果。

**返回**

评估结果，一般是 一个标量 或 多个标量。


name()
'''''''''

返回 Metric 的名字，一般通过__init__构造函数传入。

**返回**

 评估的名字，string 类型。


compute()
'''''''''

此接口可以通过 PaddlePaddle 内置的算子计算 metric 的状态，可以加速 metric 的计算，为可选的高阶接口。

- 如果这个接口定义了，输入是网络的输出 **outputs** 和 标签 **labels**，定义如：`compute(output1, output2, ..., label1, label2,...)` 。
- 如果这个接口没有定义，默认的行为是直接将输入参数返回给 `update`，则其定义如：`update(output1, output2, ..., label1, label2,...)` 。

也可以参考 `compute` 。
