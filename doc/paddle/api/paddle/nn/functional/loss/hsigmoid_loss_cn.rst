.. _cn_api_nn_functional_hsigmoid_loss:

hsigmoid_loss
-------------------------------

.. py:function:: paddle.nn.functional.hsigmoid_loss(input, label, num_classes, weight, bias=None, path_table=None, path_code=None, is_sparse=False, name=None)

层次sigmoid（hierarchical sigmoid），该OP通过构建一个分类二叉树来降低计算复杂度，主要用于加速语言模型的训练过程。

该OP建立的二叉树中每个叶节点表示一个类别(单词)，每个非叶子节点代表一个二类别分类器（sigmoid）。对于每个类别（单词），都有一个从根节点到它的唯一路径，hsigmoid累加这条路径上每个非叶子节点的损失得到总损失。

相较于传统softmax的计算复杂度 :math:`O(N)` ，hsigmoid可以将计算复杂度降至 :math:`O(logN)` ，其中 :math:`N` 表示类别总数（字典大小）。

若使用默认树结构，请参考 `Hierarchical Probabilistic Neural Network Language Model <http://www.iro.umontreal.ca/~lisa/pointeurs/hierarchical-nnlm-aistats05.pdf>`_ 。

若使用自定义树结构，请将参数 ``is_custom`` 设置为True，并完成以下步骤（以语言模型为例）：

1. 使用自定义词典来建立二叉树，每个叶结点都应该是词典中的单词；

2. 建立一个dict类型数据结构，用于存储 **单词id -> 该单词叶结点至根节点路径** 的映射，即路径表 ``path_table`` 参数；

3. 建立一个dict类型数据结构，用于存储 **单词id -> 该单词叶结点至根节点路径的编码** 的映射，即路径编码 ``path_code`` 参数。 编码是指每次二分类的标签，1为真，0为假；

4. 每个单词都已经有自己的路径和路径编码，当对于同一批输入进行操作时，可以同时传入一批路径和路径编码进行运算。

参数
::::::::::
    - **input** (Tensor) - 输入Tensor。数据类型为float32或float64，形状为 ``[N, D]`` ，其中 ``N`` 为minibatch的大小，``D`` 为特征大小。
    - **label** (Tensor) - 训练数据的标签。数据类型为int64，形状为 ``[N, 1]`` 。
    - **num_classes** (int) - 类别总数(字典大小)必须大于等于2。若使用默认树结构，即当 ``path_table`` 和 ``path_code`` 都为None时 ，必须设置该参数。若使用自定义树结构，即当 ``path_table`` 和 ``path_code`` 都不为None时，它取值应为自定义树结构的非叶节点的个数，用于指定二分类的类别总数。
    - **weight** (Tensor) - 该OP的权重参数。形状为 ``[numclasses-1, D]`` ，数据类型和 ``input`` 相同。
    - **bias** (Tensor, 可选) - 该OP的偏置参数。形状为 ``[numclasses-1, 1]`` ，数据类型和 ``input`` 相同。如果设置为None，将没有偏置参数。默认值为None。
    - **path_table** (Variable，可选) – 存储每一批样本从类别（单词）到根节点的路径，按照从叶至根方向存储。 数据类型为int64，形状为 ``[N, L]`` ，其中L为路径长度。``path_table`` 和 ``path_code`` 应具有相同的形状, 对于每个样本i，path_table[i]为一个类似np.ndarray的结构，该数组内的每个元素都是其双亲结点权重矩阵的索引。默认值为None。
    - **path_code** (Variable，可选) – 存储每一批样本从类别（单词）到根节点的路径编码，按从叶至根方向存储。数据类型为int64，形状为 ``[N, L]``。默认值为None。
    - **is_sparse** (bool，可选) – 是否使用稀疏更新方式。如果设置为True，W的梯度和输入梯度将会变得稀疏。默认值为False。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回
::::::::::
    - Tensor，层次sigmoid计算后的结果，形状为[N, 1]，数据类型和 ``input`` 一致。

代码示例
::::::::::

..  code-block:: python

    import paddle
    import paddle.nn.functional as F

    paddle.set_device('cpu')

    input = paddle.uniform([2, 3])
    # [[-0.8018668   0.8736385  -0.9064771 ] # random
    #  [-0.10228515 -0.87188244 -0.8783718 ]] # random
    label = paddle.to_tensor([0, 1, 4, 5])
    num_classes = 5
    weight=paddle.uniform([num_classes-1, 3])
    # [[-0.24148715  0.8449961  -0.7399121 ] # random
    #  [-0.9800559   0.43509364  0.9091208 ] # random
    #  [ 0.60194826  0.10430074 -0.4521166 ] # random
    #  [-0.4469818  -0.01536179 -0.604454  ]] # random

    out=F.hsigmoid_loss(input, label, num_classes, weight)
    # [[3.0159328]
    #  [2.2407534]]
