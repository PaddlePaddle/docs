.. _cn_api_fluid_metrics_MetricBase:

MetricBase
-------------------------------

.. py:class:: paddle.fluid.metrics.MetricBase(name)

所有Metrics的基类。MetricBase为模型估计方法定义一组接口。Metrics累积连续的两个minibatch之间的度量状态，对每个minibatch用最新接口将当前minibatch值添加到全局状态。用eval函数来计算last reset()或者scratch on()中累积的度量值。如果需要定制一个新的metric，请继承自MetricBase和自定义实现类。

参数：
    - **name** (str) - metric实例名。例如准确率（accuracy）。如果想区分一个模型里不同的metrics，则需要实例名。

.. py:method:: reset()

        reset()清除度量（metric）的状态（state）。默认情况下，状态（state）包含没有 ``_`` 前缀的metric。reset将这些状态设置为初始状态。如果不想使用隐式命名规则，请自定义reset接口。

.. py:method:: get_config()

获取度量（metric)状态和当前状态。状态（state）包含没有 ``_`` 前缀的成员。
        
返回：metric对应到state的字典

返回类型：字典（dict）


.. py:method:: update(preds,labels)

更新每个minibatch的度量状态（metric states），用户可通过Python或者C++操作符计算minibatch度量值（metric）。

参数：
     - **preds** (numpy.array) - 当前minibatch的预测
     - **labels** (numpy.array) - 当前minibatch的标签，如果标签为one-hot或者soft-label，应该自定义相应的更新规则。

.. py:method:: eval()

基于累积状态（accumulated states）评估当前度量（current metric）。

返回：metrics（Python中）

返回类型：float|list(float)|numpy.array







