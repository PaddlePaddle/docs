==============
metric_op
==============


.. _cn_api_fluid_layers_accuracy:

accuracy
-------------------------------

.. py:function:: paddle.fluid.layers.accuracy(input, label, k=1, correct=None, total=None)

accuracy layer。 参考 https://en.wikipedia.org/wiki/Precision_and_recall

使用输入和标签计算准确率。 每个类别中top k 中正确预测的个数。注意：准确率的 dtype 由输入决定。 输入和标签 dtype 可以不同。

参数：
    - **input** (Variable)-该层的输入，即网络的预测。支持 Carry LoD。
    - **label** (Variable)-数据集的标签。
    - **k** (int) - 每个类别的 top k
    - **correct** (Variable)-正确的预测个数。
    - **total** (Variable)-总共的样本数。

返回: 正确率

返回类型: 变量（Variable）

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(name="data", shape=[-1, 32, 32], dtype="float32")
    label = fluid.layers.data(name="label", shape=[-1,1], dtype="int32")
    predict = fluid.layers.fc(input=data, size=10)
    accuracy_out = fluid.layers.accuracy(input=predict, label=label, k=5)












.. _cn_api_fluid_layers_auc:

auc
-------------------------------

.. py:function:: paddle.fluid.layers.auc(input, label, curve='ROC', num_thresholds=4095, topk=1, slide_steps=1)

**Area Under the Curve(AUC) Layer**

该层根据前向输出和标签计算AUC，在二分类(binary classification)估计中广泛使用。

注：如果输入标注包含一种值，只有0或1两种情况，数据类型则强制转换成布尔值。相关定义可以在这里: https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve 找到

有两种可能的曲线：

1. ROC:受试者工作特征曲线

2. PR:准确率召回率曲线

参数：
    - **input** (Variable) - 浮点二维变量，值的范围为[0,1]。每一行降序排列。输入应为topk的输出。该变量显示了每个标签的概率。
    - **label** (Variable) - 二维整型变量，表示训练数据的标注。批尺寸的高度和宽度始终为1.
    - **curve** (str) - 曲线类型，可以为 ``ROC`` 或 ``PR``，默认 ``ROC``。
    - **num_thresholds** (int) - 将roc曲线离散化时使用的临界值数。默认200
    - **topk** (int) - 只有预测输出的topk数才被用于auc
    - **slide_steps** - 计算批auc时，不仅用当前步也用先前步。slide_steps=1，表示用当前步；slide_steps = 3表示用当前步和前两步；slide_steps = 0，则用所有步

返回：代表当前AUC的scalar

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
    label = fluid.layers.data(name="label", shape=[1], dtype="int32")
    predict = fluid.layers.fc(input=data, size=2)
    auc_out=fluid.layers.auc(input=prediction, label=label)








