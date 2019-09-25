.. _cn_api_fluid_layers_accuracy:

accuracy
-------------------------------

.. py:function:: paddle.fluid.layers.accuracy(input, label, k=1, correct=None, total=None)

accuracy layer。 参考 https://en.wikipedia.org/wiki/Precision_and_recall

使用输入和标签计算准确率。 如果正确的标签在topk个预测值里，则计算结果加1。注意：输出正确率的类型由input类型决定，input和lable的类型可以不一样。

参数：
    - **input** (Tensor|LoDTensor)-数据类型为float32,float64。输入为网络的预测值。
    - **label** (Tensor|LoDTensor)-数据类型为int64，int32。输入为数据集的标签。
    - **k** (int64|int32) - 取每个类别中k个预测值用于计算。
    - **correct** (int64|int32)-正确预测值的个数。
    - **total** (int64|int32)-总共的预测值。

返回: 计算出来的正确率。

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