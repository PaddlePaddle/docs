.. _cn_api_fluid_layers_cross_entropy:

cross_entropy
-------------------------------

.. py:function:: paddle.fluid.layers.cross_entropy(input, label, soft_label=False, ignore_index=-100)

该函数定义了输入和标签之间的cross entropy(交叉熵)层。该函数支持standard cross-entropy computation（标准交叉熵损失计算）
以及soft-label cross-entropy computation（软标签交叉熵损失计算）

  1. One-hot cross-entropy算法

     soft_label = False, Label[i, 0] 指明样本i的类别所具的索引:
                            .. math::
                                     \\Y[i]=-log(X[i,Label[i]])\\

  2. Soft-label cross-entropy算法

     soft_label = True, Label[i, j] 表明样本i对应类别j的soft label(软标签):
                            .. math::
                                     \\Y[i]= \sum_{j}-Label[i,j]*log(X[i,j])\\

     **请确保采用此算法时识别为各软标签的概率总和为1**

  3. One-hot cross-entropy with vecterized label（使用向量化标签的One-hot）算法

     作为 *2* 的特殊情况，当软类标签内部只有一个非零概率元素，且它的值为1，那么 *2* 算法降级为一种仅有one-hot标签的one-hot交叉熵





参数：
    - **input** (Variable|list) – 一个形为[N x D]的二维tensor，其中N是batch大小，D是类别（class）数目。 这是由之前的operator计算出的概率，绝大多数情况下是由softmax operator得出的结果
    - **label** (Variable|list) – 一个二维tensor组成的正确标记的数据集(ground truth)。 当 ``soft_label`` 为False时，label为形为[N x 1]的tensor<int64>。 ``soft_label`` 为True时, label是形为 [N x D]的 tensor<float/double>
    - **soft_label** (bool) – 标志位，指明是否需要把给定的标签列表认定为软标签。默认为False。
    - **ignore_index** (int) – 指定一个被无视的目标值，并且这个值不影响输入梯度。仅在 ``soft_label`` 为False时生效。 默认值: kIgnoreIndex

返回： 一个形为[N x 1]的二维tensor，承载了交叉熵损失

弹出异常： ``ValueError``

                        1. 当 ``input`` 的第一维和 ``label`` 的第一维不相等时，弹出异常
                        2. 当 ``soft_label`` 值为True， 且 ``input`` 的第二维和 ``label`` 的第二维不相等时，弹出异常
                        3. 当 ``soft_label`` 值为False，且 ``label`` 的第二维不是1时，弹出异常



**代码示例**

..  code-block:: python

        import paddle.fluid as fluid
        classdim = 7
        x = fluid.layers.data(name='x', shape=[3, 7], dtype='float32', append_batch_size=False)
        label = fluid.layers.data(name='label', shape=[3, 1], dtype='float32', append_batch_size=False)
        predict = fluid.layers.fc(input=x, size=classdim, act='softmax')
        cost = fluid.layers.cross_entropy(input=predict, label=label)













