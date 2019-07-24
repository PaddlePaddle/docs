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












