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

返回：代表当前AUC的一个元组
返回的元组为auc_out, batch_auc_out, [batch_stat_pos, batch_stat_neg, stat_pos, stat_neg]。

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    data = fluid.layers.data(name="input", shape=[-1, 3, 3], dtype="float32")
    label = fluid.layers.data(name="label", shape=[1], dtype="int")
    fc_out = fluid.layers.fc(input=data, size=2)
    predict = fluid.layers.softmax(input=fc_out)
    result=fluid.layers.auc(input=predict, label=label)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    exe.run(fluid.default_startup_program())
    x = np.random.rand(3, 3, 3).astype("float32")
    y = np.array([1,0,1])
    output= exe.run(feed={"input": x,"label": y},
                     fetch_list=[result[0]])
    print(output)






