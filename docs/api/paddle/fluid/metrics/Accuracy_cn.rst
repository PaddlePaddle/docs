.. _cn_api_fluid_metrics_Accuracy:

Accuracy
-------------------------------
.. py:class:: paddle.fluid.metrics.Accuracy(name=None)




该接口用来计算多个 mini-batch 的平均准确率。Accuracy 对象有两个状态 value 和 weight。Accuracy 的定义参照 https://en.wikipedia.org/wiki/Accuracy_and_precision 。

参数
::::::::::::

    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
初始化后的 ``Accuracy`` 对象

返回类型
::::::::::::
Accuracy

代码示例
::::::::::::


COPY-FROM: paddle.fluid.metrics.Accuracy

方法
::::::::::::
update(value, weight)
'''''''''

该函数使用输入的(value, weight)来累计更新 Accuracy 对象的对应状态，更新方式如下：

    .. math::
                   \\ \begin{array}{l}{\text { self. value }+=\text { value } * \text { weight }} \\ {\text { self. weight }+=\text { weight }}\end{array} \\

**参数**

    - **value** (float|numpy.array) – mini-batch 的正确率
    - **weight** (int|float) – mini-batch 的大小

**返回**
无

eval()
'''''''''

该函数计算并返回累计的 mini-batches 的平均准确率。

**返回**
累计的 mini-batches 的平均准确率

**返回类型**
float 或 numpy.array
