.. _cn_api_fluid_layers_accuracy:

accuracy
-------------------------------

.. py:function:: paddle.fluid.layers.accuracy(input, label, k=1, correct=None, total=None)

accuracy layer。 参考 https://en.wikipedia.org/wiki/Precision_and_recall

使用输入和标签计算准确率。 每个类别中top k 中正确预测的个数。注意：准确率的 dtype 由输入决定。 输入和标签 dtype 可以不同。

参数：
    - **input** (Tensor|LoDTensor)-数据类型为float32,float64。该层的输入，即网络的预测。支持 Carry LoD
    - **label** (Tensor|LoDTensor)-数据类型为int64，int32。数据集的标签
    - **k** (int64|int32) - 每个类别的 top k
    - **correct** (int64|int32)-正确的预测个数
    - **total** (int64|int32)-总共的样本数

返回: 与输入shape相同的张量

返回类型: Variable（Tensor），数据类型为float32的Tensor

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    data = fluid.layers.data(name="input", shape=[-1, 32, 32], dtype="float32")
    label = fluid.layers.data(name="label", shape=[-1,1], dtype="int")
    fc_out = fluid.layers.fc(input=data, size=10)
    predict = fluid.layers.softmax(input=fc_out)
    result = fluid.layers.accuracy(input=predict, label=label, k=5)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    exe.run(fluid.default_startup_program())
    x = np.random.rand(3, 32, 32).astype("float32")
    y = np.array([[1],[0],[1]])
    output= exe.run(feed={"input": x,"label": y},
                     fetch_list=[result[0]])
    print(output)
    
    """
    Output:
    [array([0.6666667], dtype=float32)]
    """