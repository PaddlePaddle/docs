.. _cn_api_nn_functional_hsigmoid_loss:

hsigmoid_loss
-------------------------------

.. py:function:: paddle.nn.functional.hsigmoid_loss(input, label, num_classes, weight, bias=None, path_table=None, path_code=None, is_sparse=False, name=None)

层次 sigmoid（hierarchical sigmoid），通过构建一个分类二叉树来降低计算复杂度，主要用于加速语言模型的训练过程。

建立的二叉树中每个叶节点表示一个类别（单词），每个非叶子节点代表一个二类别分类器 (sigmoid)。对于每个类别（单词），都有一个从根节点到它的唯一路径，hsigmoid 累加这条路径上每个非叶子节点的损失得到总损失。

相较于传统 softmax 的计算复杂度 :math:`O(N)` ，hsigmoid 可以将计算复杂度降至 :math:`O(logN)`，其中 :math:`N` 表示类别总数（字典大小）。

若使用默认树结构，请参考 `Hierarchical Probabilistic Neural Network Language Model <http://www.iro.umontreal.ca/~lisa/pointeurs/hierarchical-nnlm-aistats05.pdf>`_ 。

若使用自定义树结构，请将参数 ``is_custom`` 设置为 True，并完成以下步骤（以语言模型为例）：

1. 使用自定义词典来建立二叉树，每个叶结点都应该是词典中的单词；

2. 建立一个 dict 类型数据结构，用于存储 **单词 id -> 该单词叶结点至根节点路径** 的映射，即路径表 ``path_table`` 参数；

3. 建立一个 dict 类型数据结构，用于存储 **单词 id -> 该单词叶结点至根节点路径的编码** 的映射，即路径编码 ``path_code`` 参数。编码是指每次二分类的标签，1 为真，0 为假；

4. 每个单词都已经有自己的路径和路径编码，当对于同一批输入进行操作时，可以同时传入一批路径和路径编码进行运算。

参数
::::::::::
    - **input** (Tensor) - 输入 Tensor。数据类型为 float32 或 float64，形状为 ``[N, D]``，其中 ``N`` 为 minibatch 的大小，``D`` 为特征大小。
    - **label** (Tensor) - 训练数据的标签。数据类型为 int64，形状为 ``[N, 1]`` 。
    - **num_classes** (int) - 类别总数(字典大小)必须大于等于 2。若使用默认树结构，即当 ``path_table`` 和 ``path_code`` 都为 None 时，必须设置该参数。若使用自定义树结构，即当 ``path_table`` 和 ``path_code`` 都不为 None 时，它取值应为自定义树结构的非叶节点的个数，用于指定二分类的类别总数。
    - **weight** (Tensor) - 权重参数。形状为 ``[numclasses-1, D]``，数据类型和 ``input`` 相同。
    - **bias** (Tensor，可选) - 偏置参数。形状为 ``[numclasses-1, 1]``，数据类型和 ``input`` 相同。如果设置为 None，将没有偏置参数。默认值为 None。
    - **path_table** (Tensor，可选) - 存储每一批样本从类别（单词）到根节点的路径，按照从叶至根方向存储。数据类型为 int64，形状为 ``[N, L]``，其中 L 为路径长度。``path_table`` 和 ``path_code`` 应具有相同的形状，对于每个样本 i，path_table[i]为一个类似 np.ndarray 的结构，该数组内的每个元素都是其双亲结点权重矩阵的索引。默认值为 None。
    - **path_code** (Tensor，可选) - 存储每一批样本从类别（单词）到根节点的路径编码，按从叶至根方向存储。数据类型为 int64，形状为 ``[N, L]``。默认值为 None。
    - **is_sparse** (bool，可选) - 是否使用稀疏更新方式。如果设置为 True，W 的梯度和输入梯度将会变得稀疏。默认值为 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    - Tensor，层次 sigmoid 计算后的结果，形状为[N, 1]，数据类型和 ``input`` 一致。

代码示例
::::::::::

COPY-FROM: paddle.nn.functional.hsigmoid_loss
