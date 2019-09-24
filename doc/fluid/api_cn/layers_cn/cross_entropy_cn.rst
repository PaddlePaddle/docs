.. _cn_api_fluid_layers_cross_entropy:

cross_entropy
-------------------------------

.. py:function:: paddle.fluid.layers.cross_entropy(input, label, soft_label=False, ignore_index=-100)

该函数计算输入和标签之间的交叉熵。该函数支持standard cross-entropy computation（标准交叉熵损失计算）
以及soft-label cross-entropy computation（软标签交叉熵损失计算）

  1. One-hot cross-entropy算法

     soft_label = False, Label[i, 0] 表示样本i的标签索引:
                            .. math::
                                     \\Y[i]=-log(X[i,Label[i]])\\

  2. Soft-label cross-entropy算法

     soft_label = True, Label[i, j] 表明样本i对应类别j的软标签值:
                            .. math::
                                     \\Y[i]= \sum_{j}-Label[i,j]*log(X[i,j])\\

     **请确保采用此算法时各软标签的概率总和为1**

  3. One-hot cross-entropy with vecterized label（使用向量化标签的One-hot）算法

     作为 *2* 的特殊情况，当软类标签内部只有一个非零概率元素，且它的值为1，那么 *2* 算法降级为仅有一种one-hot标签的one-hot交叉熵


参数：
    - **input** (Variable|list) – 维度为 :math:[N_1, N_2, ..., N_k, D] 的多维Tensor，其中最后一维D是类别数目。数据类型为float32或float64。 
    - **label** (Variable|list) –  输入input对应的标签值。若soft_label=False，要求label维度为 :math:[N_1, N_2, ..., N_k] 或 :math:[N_1, N_2, ..., N_k, 1] ，数据类型为int64，且值必须大于等于0且小于D；若soft_label=True，要求label的维度、数据类型与input相同，且每个样本各软标签的总和为1。
    - **soft_label** (bool) – 标志位，指明是否需要把给定的标签列表认定为软标签。默认：False。
    - **ignore_index** (int) – 指定一个忽略的标签值，此标签值不参与计算，负值表示无需忽略任何标签值。仅在soft_label=False时有效。 默认：-100。

返回： 表示交叉熵结果的Tensor，数据类型与input相同。若soft_label=False，则返回值维度与label维度相同；若soft_label=True，则返回值维度为 :math:[N_1, N_2, ..., N_k, 1] 。

返回类型：Variable（Tensor）


**代码示例**：

..  code-block:: python

    import paddle.fluid as fluid    
    import numpy as np
    class_num = 7    
    x = fluid.layers.data(name='x', shape=[3, 10], dtype='float32', append_batch_size=False)    
    label = fluid.layers.data(name='label', shape=[3, 1], dtype='int64', append_batch_size=False)    
    predict = fluid.layers.fc(input=x, size=class_num, act='softmax')    
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    np_x = np.random.random(size=(3, 10)).astype('float32')
    np_label = np.random.random(size=(3, 1)).astype('int64')
    output = exe.run(feed={"x": np_x, "label": np_label}, fetch_list = [cost])
    print(output)
