.. _cn_api_paddle_nn_HSigmoidLoss:

HSigmoidLoss
-------------------------------

.. py:class:: paddle.nn.HSigmoidLoss(feature_size, num_classes, weight_attr=None, bias_attr=None, is_custom=False, is_sparse=False, name=None)

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
    - **feature_size** (int) - 输入Tensor的特征大尺寸。
    - **num_classes** (int) - 类别总数(字典大小)必须大于等于2。若使用默认树结构，即当 ``is_custom=False`` 时 ，必须设置该参数。若使用自定义树结构，即当 ``is_custom=True`` 时，它取值应为自定义树结构的非叶节点的个数，用于指定二分类的类别总数。
    - **weight_attr** (ParamAttr, 可选) – 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **bias_attr** (ParamAttr, 可选) – 指定偏置参数属性的对象，若 `bias_attr` 为bool类型，如果设置为False，表示不会为该层添加偏置；如果设置为True，表示使用默认的偏置参数属性。默认值为None，表示使用默认的偏置参数属性。默认的偏置参数属性将偏置参数的初始值设为0。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **is_custom** (bool，可选) – 是否使用用户自定义二叉树取代默认二叉树结构。如果设置为True，请务必设置 ``path_table`` ， ``path_code`` ， ``num_classes`` ，否则必须设置num_classes。默认值为False。
    - **is_sparse** (bool，可选) – 是否使用稀疏更新方式。如果设置为True，W的梯度和输入梯度将会变得稀疏。默认值为False。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

形状
:::::::::
    - **input** (Tensor): - 输入的Tensor，维度是[N, D], 其中N是batch size， D是特征尺寸。
    - **label** (Tensor): - 标签，维度是[N, 1]。
    - **output** (Tensor): - 输入 ``input`` 和标签 ``label`` 间的 `hsigmoid loss` 损失。输出Loss的维度为[N, 1]。

代码示例
::::::::::

..  code-block:: python

    import paddle
    paddle.set_device('cpu')

    input = paddle.uniform([2, 3])
    # [[-0.2820413   0.9528898  -0.81638825] # random
    #  [-0.6733154  -0.33866507  0.25770962]] # random
    label = paddle.to_tensor([0, 1, 4, 5])
    m = paddle.nn.HSigmoidLoss(3, 5)
    out = m(input, label)
    # [[2.4543471]
    #  [1.9359267]]
