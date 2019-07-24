.. _cn_api_fluid_layers_hsigmoid:

hsigmoid
-------------------------------

.. py:function:: paddle.fluid.layers.hsigmoid(input, label, num_classes, param_attr=None, bias_attr=None, name=None, path_table=None, path_code=None, is_custom=False, is_sparse=False)

层次sigmod（ hierarchical sigmoid ）加速语言模型的训练过程。这个operator将类别组织成一个完全二叉树，也可以使用 ``is_custom`` 参数来传入自定义的树结构来实现层次化。

树中每个叶节点表示一个类(一个单词)，每个内部节点进行一个二分类。对于每个单词，都有一个从根到它的叶子节点的唯一路径，hsigmoid计算路径上每个内部节点的损失（cost），并将它们相加得到总损失（cost）。

hsigmoid可以把时间复杂度 :math:`O(N)` 优化到 :math:`O(logN)` ,其中 :math:`N` 表示单词字典的大小。

使用默认树结构，请参考 `Hierarchical Probabilistic Neural Network Language Model <http://www.iro.umontreal.ca/~lisa/pointeurs/hierarchical-nnlm-aistats05.pdf>`_ 。

若要使用自定义树结构，请设置 ``is_custom`` 值为True。但在此之前，请完成以下几步：

1.使用自定义词典来建立二叉树，每个叶结点都应该是词典中的单词

2.建立一个dict类型数据结构，来存储 **单词id -> 该单词叶结点至根结点路径** 的映射，称之为路径表 ``path_table`` 参数

3.建立一个dict类型数据结构，来存储 **单词id -> 该单词叶结点至根结点路径的编码(code)** 的映射。 编码code是指每次二分类的标签，1为真，0为假

4.现在我们的每个单词都已经有自己的路径和路径编码，当对于同一批输入进行操作时，你可以同时传入一批路径和路径编码进行运算。

参数:
    - **input** (Variable) - 输入张量，shape为 ``[N×D]`` ,其中 ``N`` 是minibatch的大小，D是特征大小。
    - **label** (Variable) - 训练数据的标签。该tensor的shape为 ``[N×1]``
    - **num_classes** (int) - 类别的数量不能少于2。若使用默认树结构，该参数必须用户设置。当 ``is_custom=False`` 时，该项绝不能为None。反之，如果 ``is_custom=True`` ，它取值应为非叶节点的个数，来指明二分类实用的类别数目。
    - **param_attr** (ParamAttr|None) - 可学习参数/ hsigmoid权重的参数属性。如果将其设置为ParamAttr的一个属性或None，则将ParamAttr设置为param_attr。如果没有设置param_attr的初始化器，那么使用用Xavier初始化。默认值:没None。
    - **bias_attr** (ParamAttr|bool|None) - hsigmoid偏置的参数属性。如果设置为False，则不会向输出添加偏置。如果将其设置ParamAttr的一个属性或None，则将ParamAttr设置为bias_attr。如果没有设置bias_attr的初始化器，偏置将初始化为零。默认值:None。
    - **name** (str|None) - 该layer的名称(可选)。如果设置为None，该层将被自动命名。默认值:None。
    - **path_table** (Variable|None) – 存储每一批样本从词到根节点的路径。路径应为从叶至根方向。 ``path_table`` 和 ``path_code`` 应具有相同的形, 对于每个样本 i ，path_table[i]为一个类似np.array的结构，该数组内的每个元素都是其双亲结点权重矩阵的索引
    - **path_code** (Variable|None) – 存储每批样本的路径编码，仍然是按从叶至根方向。各样本路径编码批都由其各祖先结点的路径编码组成
    - **is_custom** (bool|False) – 使用用户自定义二叉树取代默认二叉树结构，如果该项为真， 请务必设置 ``path_table`` , ``path_code`` , ``num_classes`` , 否则就需要设置 num_classes
    - **is_sparse** (bool|False) – 使用稀疏更新方式，而非密集更新。如果为真， W的梯度和输入梯度将会变得稀疏

返回:  (LoDTensor) 层次sigmod（ hierarchical sigmoid） 。shape[N, 1]

返回类型:  Out


**代码示例**

..  code-block:: python

      import paddle.fluid as fluid
      x = fluid.layers.data(name='x', shape=[2], dtype='float32')
      y = fluid.layers.data(name='y', shape=[1], dtype='int64')
      out = fluid.layers.hsigmoid(input=x, label=y, num_classes=6)




