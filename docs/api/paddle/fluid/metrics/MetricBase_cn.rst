.. _cn_api_fluid_metrics_MetricBase:

MetricBase
-------------------------------

.. py:class:: paddle.fluid.metrics.MetricBase(name)




在评估神经网络效果的时候，由于我们常常需要把测试数据切分成mini-batch，并逐次将每个mini-batch送入神经网络进行预测和评估，因此我们每次只能获得当前batch下的评估结果，而并不能一次性获得整个测试集的评估结果。paddle.fluid.metrics正是为了解决这些问题而设计的，大部分paddle.fluid.metrics下的类都具有如下功能：

1. 接受模型对一个batch的预测结果（numpy.array）和这个batch的原始标签（numpy.array）作为输入，并进行特定的计算（如计算准确率，召回率等）。

2. 将当前batch评估结果和历史评估结果累计起来，以获取目前处理过的所有batch的整体评估结果。

MetricBase是所有paddle.fluid.metrics下定义的所有python类的基类，它定义了一组接口，并需要所有继承他的类实现具体的计算逻辑，包括：

1. update(preds, labels)：给定当前计算当前batch的预测结果（preds）和标签（labels），计算这个batch的评估结果。

2. eval()：合并当前累积的每个batch的评估结果，并返回整体评估结果。

3. reset()：清空累积的每个batch的评估结果。

方法
::::::::::::
__init__(name)
'''''''''

构造函数，参数name表示当前创建的评估器的名字。

**参数**

    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

**返回**
一个python对象，表示一个具体的评估器。

**返回类型**
python对象

reset()
'''''''''

空累积的每个batch的评估结果。

**返回**
无

update(preds,labels)
'''''''''

给定当前计算当前batch的预测结果（preds）和标签（labels），计算这个batch的评估结果，并将这个评估结果在评估器内部记录下来，注意update函数并不会返回评估结果。

**参数**

     - **preds** (numpy.array) - 当前minibatch的预测结果。
     - **labels** (numpy.array) - 当前minibatch的标签。

**返回**
无

eval()
'''''''''

合并当前累积的每个batch的评估结果，并返回整体评估结果。

**返回**
当前累积batch的整体评估结果。

**返回类型**
float|list(float)|numpy.array

get_config()
'''''''''

获取当前评估器的状态，特指评估器内部没有 ``_`` 前缀的所有成员变量。

**返回**
一个python字典，包含了当前评估器内部的状态。

**返回类型**
python字典（dict）

