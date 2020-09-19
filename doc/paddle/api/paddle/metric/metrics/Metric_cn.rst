.. _cn_api_metric_Metric:

Metric
-------------------------------

.. py:class:: paddle.metric.Metric()


评估器metric的基类。

用法:
    
    .. code-block:: text

        m = SomeMetric()
        for prediction, label in ...:
            m.update(prediction, label)
        m.accumulate()
    
:code:`compute` 接口的进阶用法:

在 :code:`compute` 中可以使用PaddlePaddle内置的算子进行评估器的状态，而不是通过
Python/NumPy, 这样可以加速计算。:code:`update` 接口将 :code:`compute` 的输出作为
输入，内部采用Python/NumPy计算。

:code: `Metric` 计算流程如下 （在{}中的表示模型和评估器的计算）:

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

**代码示例**

以 计算正确率的 :code: `Accuracy` 为例，该评估器的输入为 :code: `pred` 和
:code: `label`, 可以在 :code:`compute` 中通过 :code: `pred` 和 :code: `label`
先计算正确预测的矩阵。 例如，预测结果包含10类，:code: `pred` 的shape是[N, 10]，
:code: `label` 的shape是[N, 1], N是batch size，我们需要计算top-1和top-5的准
确率，可以在:code: `compute` 中计算每个样本的top-5得分，正确预测的矩阵的shape
是[N, 5].

        
    .. code-block:: python
        def compute(pred, label):
            # sort prediction and slice the top-5 scores
            pred = paddle.argsort(pred, descending=True)[:, :5]
            # calculate whether the predictions are correct
            correct = pred == label
            return paddle.cast(correct, dtype='float32')

在:code:`compute` 中的计算，使用内置的算子(可以跑在GPU上，是的速度更快)。
作为:code:`update` 的输入，该接口计算如下: 

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

.. py:function:: reset()

清空状态和计算结果。

返回：无


.. py:function:: update(*args)


更新状态。如果定义了:code:`compute` ，:code:`update` 的输入是:code:`compute` 的输出。
如果没有定义，则输入是网络的输出**output**和标签**label**，
如: :code:`update(output1, output2, ..., label1, label2,...)` .

也可以参考 :code:`update` 。


.. py:function:: accumulate()

累积的统计指标，计算和返回评估结果。

返回：评估结果，一般是个标量 或 多个标量。


.. py:function:: name()

返回Metric的名字, 一般通过__init__构造函数传入。

返回: 评估的名字，string类型。


.. py:function:: compute()

此接口可以通过PaddlePaddle内置的算子计算metric的状态，可以加速metric的计算，
为可选的高阶接口。

如果这个接口定义了，输入是网络的输出 **outputs** 和 标签 **labels** , 定义如:
:code:`compute(output1, output2, ..., label1, label2,...)` 。
如果这个接口没有定义, 默认的行为是直接将输入参数返回给 :code: `update` ，则其
定义如: :code:`update(output1, output2, ..., label1, label2,...)` 。

也可以参考 :code:`compute` 。
