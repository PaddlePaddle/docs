############
模型评估
############

模型评估是指用评价函数(metrics)来评估模型的好坏，可作为在训练中调整超参数、评估模型效果的重要依据。不同类型的模型任务会选取不同评价函数，常见的如回归类任务会用均方差(MSE)，二分类任务会用AUC (Area Under Curve)值等。

评价函数和loss函数非常相似，但不参与模型的训练优化。
 

评价函数的输入为模型的预测值(preds)和标注值(labels)，并返回计算后的评价指标。

paddle.fluid.metrics模块提供了一系列常用的模型评价指标; 用户也可以通过Python接口定制评价指标，或者通过定制C++ Operator的方式，在GPU上加速评价指标的计算。

常用指标
############

不同类型的任务，会选用不同的评价指标。
 
回归问题通常会用RMSE(均方根误差)，MAE(平均绝对误差)，R-Square(R平方)等
AUC(Area Under Cure)指标则常被用在分类任务(classification)上
目标检测任务(Object Detection)则经常会用到mAP(Mean Average Precision) 
 
paddle.fluid.metrics中包含了一些常用分类指标，例如Precision, Recall, Accuracy等 

下面是使用Precision指标的示例:

.. code-block:: python

   import paddle.fluid as fluid
   label = fluid.layers.data(name="label", shape=[1], dtype="int32")
   data = fluid.layers.data(name="data", shape=[32, 32], dtype="int32")
   pred = fluid.layers.fc(input=data, size=1000, act="tanh")
   acc = fluid.metrics.Precision()
   for pass_iter in range(PASSES):
     acc.reset()
     for data in train_reader():
         loss, preds, labels = exe.run(fetch_list=[cost, pred, label])
         acc.update(preds=preds, labels=labels)
     numpy_acc = acc.eval()


自定义指标
############
Fluid支持自定义指标，可灵活支持各类计算任务。下面是一个自定义的简单计数器评价函数示例:

其中preds是模型预测值，labels是标注值。

.. code-block:: python

   class MyMetric(MetricBase):
       def __init__(self, name=None):
           super(MyMetric, self).__init__(name)
           self.counter = 0  # simple counter

       def reset(self):
           self.counter = 0

       def update(self, preds, labels):
           if not _is_numpy_(preds):
               raise ValueError("The 'preds' must be a numpy ndarray.")
           if not _is_numpy_(labels):
               raise ValueError("The 'labels' must be a numpy ndarray.")
           self.counter += sum(preds == labels)

       def eval(self):
           return self.counter
