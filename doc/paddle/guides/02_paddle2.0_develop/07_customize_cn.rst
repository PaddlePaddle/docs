.. _cn_doc_customize:

自定义指标
===================

除了使用飞桨框架内置的指标外，飞桨框架还支持用户根据自己的实际场景，完成指标的自定义。


1. 自定义Loss
-----------------------

有时我们会遇到特定任务的Loss计算方式在框架既有的Loss接口中不存在，或算法不符合自己的需求，那么期望能够自己来进行Loss的自定义，我们这里就会讲解介绍一下如何进行Loss的自定义操作，首先来看下面的代码：

.. code:: ipython3
   
    class SelfDefineLoss(paddle.nn.Layer):
       """
       1. 继承paddle.nn.Layer
       """
       def __init__(self):
           """
           2. 构造函数根据自己的实际算法需求和使用需求进行参数定义即可
           """
           super(SelfDefineLoss, self).__init__()
       
       def forward(self, input, label):
           """
           3. 实现forward函数，forward在调用时会传递两个参数：input和label
               - input：单个或批次训练数据经过模型前向计算输出结果
               - label：单个或批次训练数据对应的标签数据
               接口返回值是一个Tensor，根据自定义的逻辑加和或计算均值后的损失
           """
           # 使用Paddle中相关API自定义的计算逻辑
           # output = xxxxx
           # return output
那么了解完代码层面如果编写自定义代码后我们看一个实际的例子，下面是在图像分割示例代码中写的一个自定义Loss，当时主要是想使用自定义的softmax计算维度。

.. code:: python
   
    class SoftmaxWithCrossEntropy(paddle.nn.Layer):
       def __init__(self):
           super(SoftmaxWithCrossEntropy, self).__init__()
       
        def forward(self, input, label):
           loss = F.softmax_with_cross_entropy(input,
                                               label,
                                               return_softmax=False,
                                               axis=1)
           return paddle.mean(loss)


2. 自定义Metric
----------------------------

和Loss一样，如果遇到一些想要做个性化实现的操作时，我们也可以来通过框架完成自定义的评估计算方法，具体的实现方式如下：

.. code:: ipython3

   class SelfDefineMetric(paddle.metric.Metric):
       """
       1. 继承paddle.metric.Metric
       """
       def __init__(self):
           """
           2. 构造函数实现，自定义参数即可
           """
           super(SelfDefineMetric, self).__init__()
       
       def name(self):
           """
           3. 实现name方法，返回定义的评估指标名字
           """
           return '自定义评价指标的名字'
       
       def compute(self, ...)
           """
           4. 本步骤可以省略，实现compute方法，这个方法主要用于`update`的加速，可以在这个方法中调用一些paddle实现好的Tensor计算API，编译到模型网络中一起使用低层C++ OP计算。
           """
           return 自己想要返回的数据，会做为update的参数传入。
       
       def update(self, ...):
           """
           5. 实现update方法，用于单个batch训练时进行评估指标计算。
           - 当`compute`类函数未实现时，会将模型的计算输出和标签数据的展平作为`update`的参数传入。
           - 当`compute`类函数做了实现时，会将compute的返回结果作为`update`的参数传入。
           """
           return acc value
       
       def accumulate(self):
           """
           6. 实现accumulate方法，返回历史batch训练积累后计算得到的评价指标值。
           每次`update`调用时进行数据积累，`accumulate`计算时对积累的所有数据进行计算并返回。
           结算结果会在`fit`接口的训练日志中呈现。
           """
           # 利用update中积累的成员变量数据进行计算后返回
           return accumulated acc value
       
       def reset(self):
           """
           7. 实现reset方法，每个Epoch结束后进行评估指标的重置，这样下个Epoch可以重新进行计算。
           """
           # do reset action

我们看一个框架中的具体例子，这个是框架中已提供的一个评估指标计算接口，这里就是按照上述说明中的实现方法进行了相关类继承和成员函数实现。

.. code:: ipython3
   
    from paddle.metric import Metric
    
    class Precision(Metric):
        """
        Precision (also called positive predictive value) is the fraction of
        relevant instances among the retrieved instances. Refer to
        https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers
        Noted that this class manages the precision score only for binary
        classification task.
        
        ......
        """
       
        def __init__(self, name='precision', *args, **kwargs):
            super(Precision, self).__init__(*args, **kwargs)
            self.tp = 0  # true positive
            self.fp = 0  # false positive
            self._name = name
       
        def update(self, preds, labels):
            """
            Update the states based on the current mini-batch prediction results.
            Args:
                preds (numpy.ndarray): The prediction result, usually the output
                   of two-class sigmoid function. It should be a vector (column
                   vector or row vector) with data type: 'float64' or 'float32'.
               labels (numpy.ndarray): The ground truth (labels),
                   the shape should keep the same as preds.
                   The data type is 'int32' or 'int64'.
            """
            if isinstance(preds, paddle.Tensor):
                preds = preds.numpy()
            elif not _is_numpy_(preds):
                raise ValueError("The 'preds' must be a numpy ndarray or Tensor.")
            if isinstance(labels, paddle.Tensor):
                labels = labels.numpy()
            elif not _is_numpy_(labels):
                raise ValueError("The 'labels' must be a numpy ndarray or Tensor.")
           
            sample_num = labels.shape[0]
            preds = np.floor(preds + 0.5).astype("int32")
           
            for i in range(sample_num):
                pred = preds[i]
                label = labels[i]
                if pred == 1:
                    if pred == label:
                        self.tp += 1
                    else:
                        self.fp += 1
       
        def reset(self):
            """
            Resets all of the metric state.
            """
            self.tp = 0
            self.fp = 0
        
        def accumulate(self):
            """
            Calculate the final precision.

            Returns:
               A scaler float: results of the calculated precision.
            """
            ap = self.tp + self.fp
            return float(self.tp) / ap if ap != 0 else .0
        
        def name(self):
            """
            Returns metric name
            """
            return self._name


3. 自定义Callback
-------------------------------

``fit``\ 接口的callback参数支持我们传一个Callback类实例，用来在每轮训练和每个batch训练前后进行调用，可以通过callback收集到训练过程中的一些数据和参数，或者实现一些自定义操作。

.. code:: ipython3
   
    class SelfDefineCallback(paddle.callbacks.Callback):
        """
        1. 继承paddle.callbacks.Callback
        2. 按照自己的需求实现以下类成员方法：
            def on_train_begin(self, logs=None)                 训练开始前，`Model.fit`接口中调用
            def on_train_end(self, logs=None)                   训练结束后，`Model.fit`接口中调用
            def on_eval_begin(self, logs=None)                  评估开始前，`Model.evaluate`接口调用
            def on_eval_end(self, logs=None)                    评估结束后，`Model.evaluate`接口调用
            def on_test_begin(self, logs=None)                  预测测试开始前，`Model.predict`接口中调用
            def on_test_end(self, logs=None)                    预测测试结束后，`Model.predict`接口中调用
            def on_epoch_begin(self, epoch, logs=None)          每轮训练开始前，`Model.fit`接口中调用
            def on_epoch_end(self, epoch, logs=None)            每轮训练结束后，`Model.fit`接口中调用
            def on_train_batch_begin(self, step, logs=None)     单个Batch训练开始前，`Model.fit`和`Model.train_batch`接口中调用
            def on_train_batch_end(self, step, logs=None)       单个Batch训练结束后，`Model.fit`和`Model.train_batch`接口中调用
            def on_eval_batch_begin(self, step, logs=None)      单个Batch评估开始前，`Model.evalute`和`Model.eval_batch`接口中调用
            def on_eval_batch_end(self, step, logs=None)        单个Batch评估结束后，`Model.evalute`和`Model.eval_batch`接口中调用
            def on_test_batch_begin(self, step, logs=None)      单个Batch预测测试开始前，`Model.predict`和`Model.test_batch`接口中调用
            def on_test_batch_end(self, step, logs=None)        单个Batch预测测试结束后，`Model.predict`和`Model.test_batch`接口中调用
        """
        
        def __init__(self):
            super(SelfDefineCallback, self).__init__()
        # 按照需求定义自己的类成员方法


我们看一个框架中的实际例子，这是一个框架自带的ModelCheckpoint回调函数，方便用户在fit训练模型时自动存储每轮训练得到的模型。

.. code:: python
    
    class ModelCheckpoint(Callback):
        def __init__(self, save_freq=1, save_dir=None):
            self.save_freq = save_freq
            self.save_dir = save_dir
       
        def on_epoch_begin(self, epoch=None, logs=None):
            self.epoch = epoch
       
        def _is_save(self):
            return self.model and self.save_dir and ParallelEnv().local_rank == 0
        
        def on_epoch_end(self, epoch, logs=None):
            if self._is_save() and self.epoch % self.save_freq == 0:
                path = '{}/{}'.format(self.save_dir, epoch)
                print('save checkpoint at {}'.format(os.path.abspath(path)))
                self.model.save(path)
        
        def on_train_end(self, logs=None):
            if self._is_save():
                path = '{}/final'.format(self.save_dir)
                print('save checkpoint at {}'.format(os.path.abspath(path)))
                self.model.save(path)
