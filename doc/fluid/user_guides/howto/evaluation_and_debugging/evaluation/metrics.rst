############
模型评估
############

模型评估是指用评价指标(metrics)反映模型在预期目标下的精度，可作为在训练中调整超参数、评估模型效果的重要依据。其中，评价指标根据模型任务决定，也被称为评价函数。评价函数和loss函数非常相似，但不参与模型的训练优化。

评价函数的输入为模型的预测值(preds)和真实值(labels)，返回评价值。
paddle.fluid.metrics模块提供了一系列常用的模型评价指标; 用户也可以方便的通过Python定制评价指标，或者是通过定制C++ Operator的方式，在GPU上加速评价指标的计算。

常用指标
############

根据不同的任务，会选用不同的评价指标。

回归任务labels是实数，评价指标可参考 MSE (Mean Squared Error) 方法。
分类任务常用指标为分类指标（classification metrics），本文提到的一般是二分类指标，多分类(multi-category)和多标签(multi-label)任务的评价指标需要查看对应的API文档。例如排序指标AUC可以同时用在二分类和多分类任务中，因为多分类任务可以转化为二分类任务。
Fluid中包含了常用分类指标，例如Precision, Recall, Accuracy等,更多请阅读API文档。以 :ref:`Precision` 为例，具体方法为

.. code-block:: python

   >>> import paddle.fluid as fluid
   >>> labels = fluid.layers.data(name="data", shape=[1], dtype="int32")
   >>> data = fluid.layers.data(name="data", shape=[32, 32], dtype="int32")
   >>> pred = fluid.layers.fc(input=data, size=1000, act="tanh")
   >>> acc = fluid.metrics.Precision()
   >>> for pass in range(PASSES):
   >>>   acc.reset()
   >>>   for data in train_reader():
   >>>       loss, preds, labels = exe.run(fetch_list=[cost, preds, labels])
   >>>   acc.update(preds=preds, labels=labels)
   >>>   numpy_acc = acc.eval()
      

其他任务例如MultiTask Learning，Metric Learning，Learning To Rank各种指标构造方法请参考API文档。

自定义指标
############
Fluid支持自定义指标，灵活支持各类计算任务。下文通过一个简单的计数器metric函数，实现对模型的评估。
其中preds是模型预测值，labels是给定的标签。

.. code-block:: python

   >>> class MyMetric(MetricBase):
   >>>     def __init__(self, name=None):
   >>>         super(MyMetric, self).__init__(name)
   >>>         self.counter = 0  # simple counter

   >>>     def reset(self):
   >>>         self.counter = 0

   >>>     def update(self, preds, labels):
   >>>         if not _is_numpy_(preds):
   >>>             raise ValueError("The 'preds' must be a numpy ndarray.")
   >>>         if not _is_numpy_(labels):
   >>>             raise ValueError("The 'labels' must be a numpy ndarray.")
   >>>         self.counter += sum(preds == labels)

   >>>     def eval(self):
   >>>         return self.counter
