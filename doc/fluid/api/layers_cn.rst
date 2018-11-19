
.. _cn_api_fluid_layers_While:

While
>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.While (cond, is_test=False, name=None)


该类用于实现while循环控制功能。


参数：  
		- **cond** (Variable) – 用于比较的条件
		- **is_test** (bool) – 用于表明是不是在测试阶段执行
		- **name** (str) - 该层的命名
 
**代码示例**

..  code-block:: python

  d0 = layers.data("d0", shape=[10], dtype='float32')
  data_array = layers.array_write(x=d0, i=i)
  array_len = layers.fill_constant(shape=[1],dtype='int64', value=3)
  cond = layers.less_than(x=i, y=array_len)
  while_op = layers.While(cond=cond)
  with while_op.block():
      d = layers.array_read(array=data_array, i=i)
      i = layers.increment(x=i, in_place=True)
      layers.array_write(result, i=i, array=d)
      layers.less_than(x=i, y=array_len, cond=cond)

.. _cn_api_fluid_layers_Switch:

Switch
>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.Switch (name=None)

Switch类实现的功能十分类似if-elif-else。它可以在学习率调度器(learning rate scheduler)中调整学习率。
:: 
  语义上，
      1. switch控制流挨个检查cases
      2. 各个case的条件是一个布尔值(boolean)，它是一个标量(scalar)变量
      3. 它将执行第一个匹配的case后面的分支，如果没有匹配的case，但若存在一个default case,则会执行default case后面的语句
      4. 一旦匹配了一个case,它降会执行这个case所对应的分支，且仅此分支。

**代码示例**

..  code-block:: python
    
    lr = fluid.layers.tensor.create_global_var(
        shape=[1],
        value=0.0,
        dtype='float32',
        persistable=True,
        name="learning_rate")
    one_var = tensor.fill_constant(
        shape=[1], dtype='float32', value=1.0)
    two_var = tensor.fill_constant(
        shape=[1], dtype='float32', value=2.0)

    with fluid.layers.control_flow.Switch() as switch:
        with switch.case(global_step == zero_var):
            fluid.layers.tensor.assign(input=one_var, output=lr)
        with switch.default():
            fluid.layers.tensor.assign(input=two_var, output=lr)
 
.. py:method:: case(condition)

为该condition（情况，条件）建立新的block（块）。
  
  
.. py:method:: default()

为该switch建立default case。




.. _cn_api_fluid_layers_increment:
  
increment
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  
.. py:class:: paddle.fluid.layers.increment(x, value=1.0, in_place=True)

   
该函数为x中的每一个值增加 ``value`` 大小, ``value`` 即函数中待传入的参数。该函数默认直接在原变量x上进行运算。
  
参数:
    - **x** (Variable|list) – 含有输入值的张量(tensor)
    - **value** (float) – 需要增加在x变量上的值
    - **in_place** (bool) – 是否在x变量本身进行增加操作，而非返回其增加后的一个副本本身不改变。默认为True, 即在其本身进行操作。

返回： 每个元素增加后的对象
返回类型：变量(variable)

**代码示例**

..  code-block:: python
  
    data = fluid.layers.data(name='data', shape=[32, 32], dtype='float32')
    data = fluid.layers.increment(x=data, value=3.0, in_place=True)
 
 
 
.. _cn_api_fluid_layers_array_write:
    
array_write
>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.array_write(x, i, array=None)


该函数将给定的输入变量（即 ``x`` ）写入一个作为输出的 ``LOD_TENSOR_ARRAY`` 变量的某一指定位置中，
这一位置由数组下标(即 ``i`` )指明。 如果 ``LOD_TENSOR_ARRAY`` (即 ``array`` )未指定（即为None值）， 一个新的 ``LOD_TENSOR_ARRAY`` 将会被创建并作为结果返回。

参数:
    - **x** (Variable|list) – 待从中读取数据的输入张量(tensor)
    - **i** (Variable|list) – 输出结果 ``LOD_TENSOR_ARRAY`` 的下标, 该下标指向输入张量 ``x`` 写入输出数组的位置
    - **array** (Variable|list) – 会被输入张量 ``x`` 写入的输出结果 ``LOD_TENSOR_ARRAY`` 。如果该项值为None, If this parameter is NONE, 一个新的 ``LOD_TENSOR_ARRAY`` 将会被创建并作为结果返回
 
返回:	输入张量 ``x`` 所写入的输出结果 ``LOD_TENSOR_ARRAY``  
返回类型:	变量（Variable）

**代码示例**

..  code-block:: python

  tmp = fluid.layers.zeros(shape=[10], dtype='int32')
  i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
  arr = layers.array_write(tmp, i=i)



.. _cn_api_fluid_layers_crf_decoding:

crf_decoding
>>>>>>>>>>>>>>>>>>>>

.. py:class::  paddle.fluid.layers.crf_decoding(input, param_attr, label=None)

该函数读取由 ``linear_chain_crf`` 学习的emission feature weights（发射状态特征的权重）和 transition feature weights(转移特征的权重)。
本函数实现了Viterbi算法，可以动态地寻找隐藏状态最可能的序列，该序列也被称为Viterbi路径（Viterbi path），即为观察得出的标注(tags)序列。

这个运算的结果会随着 ``Label`` 参数的有无而改变：
      
      1. ``Label`` 非None的情况，在实际训练中时常发生。此时本函数会协同 ``chunk_eval`` 工作。本函数会返回一行形为[N X 1]的向量，其中值为0的部分代表该label不适合作为对应结点的标注，值为1的部分则反之。此类型的输出可以直接作为 ``chunk_eval`` 算子的输入
      
      2. 当没有 ``Label`` 时，该函数会执行标准decoding过程

（没有 ``Label`` 时）该运算返回一个形为 [N X 1]的向量，其中元素取值范围为 0 ~ 最大标注个数-1，分别为预测出的标注（tag）所在的索引。
	
参数：	
    - **input** (Variable)(LoDTensor，默认类型为 LoDTensor<float>) -一个形为 [N x D] 的LoDTensor，其中 N 是mini-batch的大小，D是标注（tag) 的总数。 该输入是 ``linear_chain_crf`` 的 unscaled emission weight matrix （未标准化的发射权重矩阵）
    - **param_attr** (ParamAttr) - 参与训练的参数的属性
    - **label** (Variable)(LoDTensor，默认类型为 LoDTensor<int64_t>) - 形为[N x 1]的正确标注（ground truth）。 该项可选择传入。 有关该参数的更多信息，请详见上述描述

返回：(LoDTensor, LoDTensor<int64_t>)decoding结果。具体内容根据 ``Label`` 参数是否提供而定。请参照函数介绍来详细了解。

返回类型： Variable


**代码示例**

..  code-block:: python

      crf_decode = layers.crf_decoding(
           input=hidden, param_attr=ParamAttr(name="crfw"))




.. _cn_api_fluid_layers_cross_entropy:

cross_entropy
>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.cross_entropy(input, label, soft_label=False, ignore_index=-100)

该函数定义了输入和标签之间的cross entropy(交叉熵)层。该函数支持standard cross-entropy computation（标准交叉熵损失函数计算）
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
    - **ignore_index** (int) – 指定一个被无视的目标值，并且这个值对不影响输入梯度变化。仅在 ``soft_label`` 为False时生效。 默认值: -100

返回： 一个形为[N x 1]的二维tensor，承载了交叉熵损失

弹出异常： ``ValueError`` 

                        1. 当 ``input`` 的第一维和 ``label`` 的第一维不相等时，弹出异常
                        2. 当 ``soft_label`` 值为True， 且 ``input`` 的第二维和 ``label`` 的第二维不相等时，弹出异常
                        3. 当 ``soft_label`` 值为False，且 ``label`` 的第二维不是1时，弹出异常
                        


**代码示例**

..  code-block:: python

        predict = fluid.layers.fc(input=net, size=classdim, act='softmax')
        cost = fluid.layers.cross_entropy(input=predict, label=label)





.. _cn_api_fluid_layers_less_than:

less_than
>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.less_than(x, y, force_cpu=None, cond=None, **ignored)


该函数按元素出现顺序依次在X,Y上操作，并返回 ``Out`` ，它们三个都是n维tensor（张量）。
其中，X、Y可以是任何类型的tensor，Out张量的各个元素可以通过 *Out=X<Y* 计算得出。

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    less = fluid.layers.less_than(x=label, y=limit)

参数：  
    - **x** (Variable) – ``less_than`` 运算的左操作数
    - **y** (Variable) – ``less_than`` 运算的右操作数
    - **force_cpu** (BOOLEAN) – 值True则强制将输出变量写入CPU内存中。否则，将其写入目前所在的运算设备上。默认为True
    - **cond** (Variable|None) – 可选的用于存储 ``less_than`` 输出结果的变量，为None则由函数自动生成Out变量


返回：	n维bool型tensor，其中各个元素可以通过 *Out=X<Y* 计算得出





.. _cn_api_fluid_layers_reorder_lod_tensor_by_rank:

reorder_lod_tensor_by_rank
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.reorder_lod_tensor_by_rank(x, rank_table)


函数参数 ``X`` 是由多个序列(sequence)组成的的一个batch（数据批）。``rank_table`` 存储着batch中序列的重新排列规则。
该operator（算子）根据 ``rank_table`` 中提供的规则信息来实现对 ``X`` 的重新排列。


::
	
  例如:
 
  假设在 RankTable 中存储的序列索引为 [3,0,2,1]， X 将会被这样被重新排列：
  X 中的第四个序列（即索引为3的序列，后面以此类推）会变成排列后的batch中的第一个，紧接着就是原来batch中的第一个元素，第三个元素，和第二个元素。
  简言之，若有原batch：X = [Seq0, Seq1, Seq2, Seq3] 且 RankTable 中的索引为 [3,0,2,1]，那么输出即为 Out = [Seq3, Seq0, Seq2, Seq1] ，它携带着新的LoD信息。	
  如果 X 的LoD信息是空的，这表明 X 不是序列型数据。这和由多个定长为1的序列组成的batch是相同的情况。此时，该函数将对 X 中的切片（slice） 按 rank_table 里的规则加以排列。
  例如，现有 X = [Slice0, Slice1, Slice2, Slice3] ，并且它LoD信息为空，在 RankTable 索引为[3, 0, 2, 1]。则 Out = [Slice3, Slice0, Slice2, Slice1] ，并且不在其中追加LoD信息。

注意，该operator对 ``X`` 进行的排序所依据的 ``LoDRankTable`` 不一定是在 ``X`` 的基础上得出来的。它可以由
其他不同的序列batch得出，并由该operator依据这个 ``LoDRankTable`` 来对  ``X`` 排序。

参数：   
    - **x** (LoDTensor)-待根据提供的 ``RankTable`` 进行排序的LoD tensor
    - **rank_table** (LoDRankTable)- ``X`` 重新排序的依据规则表


返回：	重新排列后的LoDTensor

返回类型:	LoDTensor






.. _cn_api_fluid_layers_fc:

fc
>>>>>>>>>>>>

.. py:class::  paddle.fluid.layers.fc(input, size, num_flatten_dims=1, param_attr=None, bias_attr=None, act=None, is_test=False, name=None)


**全连接层**

该函数在神经网络中建立一个全连接层。 它可以同时将多个tensor作为自己的输入，并为每个输入的tensor创立一个变量，称为“权”（weights），等价于一个从每个输入单元到每个输出单元的全连接权矩阵。FC层用每个tensor和它对应的权相乘得到输出tensor。如果有多个输入tensor，那么多个乘法运算将会加在一起得出最终结果。如果 ``bias_attr`` 非空，则会新创建一个偏向变量（bias variable），并把它加入到输出结果的运算中。最后，如果 ``act`` 非空，它也会加入最终输出的计算中。

这个过程可以通过如下公式表现：

.. math::

        \\Out=Act(\sum^{N-1}_{i=0}X_iW_i+b) \\


上述等式中：
  - :math:`N` ：输入tensor的数目
  - :math:`X_i` : 输入的tensor
  - :math:`W` ：该层创立的权
  - :math:`b` ：该层创立的bias参数
  - :math:`Act` : activation function(激励函数)
  - :math:`Out` : 输出tensor


参数:
  - **input** (Variable|list of Variable) – 该层的输入tensor(s)（张量），其维度至少是2
  - **size** (int) – 该层输出单元的数目
  - **num_flatten_dims** (int, default 1) – fc层可以接受一个维度大于2的tensor。此时， 它首先会被扁平化(flattened)为一个二维矩阵。 参数``num_flatten_dims`` 决定了输入tensor的flattened方式: 前 ``num_flatten_dims`` (包含边界，从1开始数) 个维度会被扁平化为最终矩阵的第一维 (维度即为矩阵的高), 剩下的 rank(X) - num_flatten_dims 维被扁平化为最终矩阵的第二维 (即矩阵的宽)。 例如， 假设X是一个五维tensor，其形可描述为[2, 3, 4, 5, 6], 且num_flatten_dims = 3。那么扁平化的矩阵形状将会如此： [2 x 3 x 4, 5 x 6] = [24, 30]
  - **param_attr** (ParamAttr|list of ParamAttr, default None) – 该层可学习的参数/权的参数属性
  - **bias_attr** (ParamAttr|list of ParamAttr, default None) – 该层bias变量的参数属性。如果值为False， 则bias变量不参与输出单元运算。 如果值为None，bias变量被初始化为0。默认为 None。
  - **act** (str, default None) – 应用于输出的Activation（激励函数）
  - **is_test** (bool) – 表明当前执行是否处于测试阶段的标志
  - **name** (str, default None) – 该层的命名


返回：转换结果

返回类型: Variable

弹出异常：``ValueError`` - 如果输入tensor的维度小于2

**代码示例**

..  code-block:: python

        data = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
        fc = fluid.layers.fc(input=data, size=1000, act="tanh")






.. _cn_api_fluid_layers_dynamic_gru:

dynamic_gru
>>>>>>>>>>>>>>>>>>>

.. py:class::  paddle.fluid.layers.dynamic_gru(input, size, param_attr=None, bias_attr=None, is_reverse=False, gate_activation='sigmoid', candidate_activation='tanh', h_0=None)



**实现了Gated Recurrent Unit层。**

详细理论介绍，请参照 `Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling`_。

.. _Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling: https://arxiv.org/abs/1412.3555


公式如下：

.. math::
  
  \\u_{t}=act_g(W_{ux}x_{t}+W_{uh}h_{t-1}+b_{u})\\
  \\r_{t}=act_g(W_{rx}x_{t}+W_{rh}h_{t-1}+b_{r})\\
  \\\widetilde{h_{t}}=act_{c}(W_{cx}x_{t}+W_{ch}(r_{t}\odot h_{t-1})+b_c)\\
  \\h_t=(1-u_t)\odot h_{t-1}+u_t\odot \widetilde{h_t}\\


其中，⊙为按元素将向量相乘。 :math:`act_g` 是更新门（update gate）和重置门（reset gate）的激励函数(activation)， 常为 :math:`sigmoid` 函数。 :math:`act_c` 是candidate hidden state(候选隐藏状态)的激励函数，常为 :math:`tanh` 。

注意 :math:`W_{ux}x_{t},W_{rx}x_{t},W_{cx}x_{t}` 这些在 input  :math:`x_t` 上的操作不包括在该运算中。用户可以选择性地在GRU层之前使用FC层来进行这一操作。



参数:
  - **input** (Variable) – dynamic_gru层的输入, 支持variable time length input sequence（可变时长输入序列）。 本变量底层的tensor是一个(T×3D)矩阵， 其中T是该mini-batch中总时间步数， D是隐藏状态的规模（hidden size）。
  - **size** (int) – GRU cell的维度
  - **param_attr** (ParamAttr|None)  –  可学习的隐藏层权重矩阵的参数属性。 
    注意：
                                    - 该矩阵为一个（T X 3D）矩阵。其中D为隐藏状态的规模（hidden size）
                                    - 该矩阵的所有元素由两部分组成。一是update gate和reset gate的权重，形为（D X 2D)，二是候选隐藏状态（candidate hidden state）的权重，形为 (D X D)
    如果该函数参数被设为None或者 ``ParamAttr`` 类的属性之一，则会生成一个 ``ParamAttr`` 类的对象作为param_attr。如果param_attr未被初始化（即其构造函数未被设置），Xavier会负责初始化它。 默认值为None。
  - **bias_attr** (ParamAttr|bool|None) - GRU层bias的参数属性。该（1 X 3D）形的bias变量将会连结（concatenate）在update gate（更新门）、reset gate（重置门）、candidate calculations（候选隐藏状态计算）后。如果值为False，将没有bias会应用到上述三个过程中。如果该函数参数被设为None或者 ``ParamAttr`` 类的属性之一， ``dynamic_gru`` 会生成一个 ``ParamAttr`` 类的对象作为param_attr。如果bias_attr未被初始化（即其构造函数未被设置），则它会被初始化为0。默认值为None。
  - **is_reverse** (bool) –是否计算反GRU(reversed GRU)，默认为False
  - **gate_activation** (str) – update gate 和 reset gate的激励函数（activation）。 可选择[“sigmoid”, “tanh”, “relu”, “identity”]其一, 默认为 “sigmoid”
  - **candidate_activation** (str) – candidate hidden state（候选隐藏状态）计算所需的激励函数（activation）。 可从[“sigmoid”, “tanh”, “relu”, “identity”]中选择, 默认为 “tanh”
  - **h_0** (Variable) – 该函数参数为初始隐藏状态。若未赋值，则默认为0。它是一个 (N x D) tensor, 其中 N 为输入mini-batch的总时间步数， D 为 隐藏状态规模(hidden size)
  
  
返回：	GRU的隐藏状态(hidden state)。形为（T X D），序列长度和输入相同。

返回类型:	变量（variable）


**代码示例**

..  code-block:: python

    dict_dim, emb_dim = 128, 64
    data = fluid.layers.data(name='sequence', shape=[1],
                             dtype='int32', lod_level=1)
    emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])
    hidden_dim = 512
    x = fluid.layers.fc(input=emb, size=hidden_dim * 3)
    hidden = fluid.layers.dynamic_gru(input=x, dim=hidden_dim)








.. _cn_api_fluid_layers_dynamic_lstm:

dynamic_lstm
>>>>>>>>>>>>>>>>>

.. py:class::  paddle.fluid.layers.dynamic_lstm(input, size, h_0=None, c_0=None, param_attr=None, bias_attr=None, use_peepholes=True, is_reverse=False, gate_activation='sigmoid', cell_activation='tanh', candidate_activation='tanh', dtype='float32', name=None)

LSTM，即Long-Short Term Memory(长短期记忆)运算。

默认实现方式为diagonal/peephole连接(https://arxiv.org/pdf/1402.1128.pdf)，公式如下：

.. math::
      \\i_t=\sigma (W_{ix}x_{t}+W_{ih}h_{t-1}+W_{ic}c_{t-1}+b_i)\\
      \\f_t=\sigma (W_{fx}x_{t}+W_{fh}h_{t-1}+W_{fc}c_{t-1}+b_f)\\
      \\\widetilde{c_t}=act_g(W_{ct}x_{t}+W_{ch}h_{t-1}+b_{c})\\
      \\o_t=\sigma (W_{ox}x_{t}+W_{oh}h_{t-1}+W_{oc}c_{t}+b_o)\\
      \\c_t=f_t\odot c_{t-1}+i_t\odot \widetilde{c_t}\\
      \\h_t=o_t\odot act_h(c_t)\\

W 代表了权重矩阵(weight matrix)，例如 :math:`W_{xi}` 是从输入门（input gate）到输入的权重矩阵, :math:`W_{ic}` ，:math:`W_{fc}` ，  :math:`W_{oc}` 是对角权重矩阵(diagonal weight matrix)，用于peephole连接。在此实现方式中，我们使用向量来代表这些对角权重矩阵。其中：
      - :math:`b` 表示bias向量（ :math:`b_i` 是输入门的bias向量）
      - :math:`σ` 是非线性激励函数（non-linear activations），比如逻辑sigmoid函数
      - :math:`i` ，:math:`f` ，:math:`o` 和 :math:`c` 分别为输入门(input gate)，遗忘门(forget gate)，输出门（output gate）,以及神经元激励向量（cell activation vector）这些向量和神经元输出激励向量（cell output activation vector）:math:`h` 有相同的大小。
      - :math:`⊙` 意为按元素将两向量相乘
      - :math:`act_g` , :math:`act_h` 分别为神经元(cell)输入、输出的激励函数(activation)。常常使用tanh函数。
      - :math:`\widetilde{c_t}` 也被称为候选隐藏状态(candidate hidden state)。可根据当前输入和之前的隐藏状态计算而得

将 ``use_peepholes`` 设为False来禁用 peephole 连接方法。 公式等详细信息请参考 http://www.bioinf.jku.at/publications/older/2604.pdf 。

注意， :math:`W_{xi}x_t, W_{xf}x_t, W_{xc}x_t,W_{xo}x_t` 这些在输入 :math:`x_t` 上的操作不包括在此运算中。用户可以在LSTM operator之前选择使用全连接运算。




参数:
  - **input** (Variable) (LoDTensor) - LodTensor类型，支持variable time length input sequence（时长可变的输入序列）。 该LoDTensor中底层的tensor是一个形为(T X 4D)的矩阵，其中T为此mini-batch上的总共时间步数。D为隐藏层的大小、规模(hidden size)
  - **size** (int) – 4 * 隐藏层大小
  - **h_0** (Variable) – 最初的隐藏状态（hidden state），可选项。默认值为0。它是一个(N x D)张量，其中N是batch大小，D是隐藏层大小。
  - **c_0** (Variable) – 最初的神经元状态（cell state）， 可选项。 默认值0。它是一个(N x D)张量, 其中N是batch大小。h_0和c_0仅可以同时为None，不能只其中一个为None。
  - **param_attr** (ParamAttr|None) – 可学习的隐藏层权重的参数属性。
    注意：
                      - Weights = :math: \{W_{ch}, W_{ih},  W_{fh},  W_{oh} \}
                      - 形为(D x 4D), 其中D是hidden size（隐藏层规模）

    如果它被设为None或者 ``ParamAttr`` 属性之一, dynamic_lstm会创建 ``ParamAttr`` 对象作为param_attr。如果没有对param_attr初始化（即构造函数没有被设置）， Xavier会负责初始化参数。默认为None。
  - **bias_attr** (ParamAttr|None) – 可学习的bias权重的属性, 包含两部分，input-hidden bias weights（输入隐藏层的bias权重）和 peephole connections weights（peephole连接权重）。如果 ``use_peepholes`` 值为 ``True`` ， 则意为使用peephole连接的权重。
    另外：
      - use_peepholes = False - Biases = :math:`\{ b_c,b_i,b_f,b_o \}` - 形为(1 x 4D)。
      - use_peepholes = True - Biases = :math:`\{ b_c,b_i,b_f,b_o,W_{ic},W_{fc},W_{oc} \}` - 形为 (1 x 7D)。

    如果它被设为None或 ``ParamAttr`` 的属性之一， ``dynamic_lstm`` 会创建一个 ``ParamAttr``对象作为bias_attr。 如果没有对bias_attr初始化（即构造函数没有被设置），bias会被初始化为0。默认值为None。
  - **use_peepholes** (bool) – （默认: True） 是否使用diagonal/peephole连接方式
  - **is_reverse** (bool) – （默认: False） 是否计算反LSTM(reversed LSTM)
  - **gate_activation** (str) – （默认: "sigmoid"）应用于input gate（输入门），forget gate（遗忘门）和 output gate（输出门）的激励函数（activation），默认为sigmoid
  - **cell_activation** (str) – （默认: tanh）用于神经元输出的激励函数(activation), 默认为tanh
  - **candidate_activation** (str) – （默认: tanh）candidate hidden state（候选隐藏状态）的激励函数(activation), 默认为tanh 
  - **dtype** (str) – 即 Data type（数据类型）。 可以选择 [“float32”, “float64”]，默认为“float32”
  - **name** (str|None) – 该层的命名，可选项。如果值为None, 将会自动对该层命名

返回：隐藏状态（hidden state），LSTM的神经元状态。两者都是（T x D）形，且LoD保持与输入一致

返回类型: 元组（tuple）


**代码示例**

..  code-block:: python

  hidden_dim = 512
  forward_proj = fluid.layers.fc(input=input_seq, size=hidden_dim * 4,
                                 bias_attr=False)
  forward, _ = fluid.layers.dynamic_lstm(
      input=forward_proj, size=hidden_dim * 4, use_peepholes=False)





.. _cn_api_fluid_layers_gru_unit:

gru_unit
>>>>>>>>>>>>

.. py:class::  paddle.fluid.layers.gru_unit(input, hidden, size, param_attr=None, bias_attr=None, activation='tanh', gate_activation='sigmoid')

GRU单元层。GRU执行步骤基于如下等式：

.. math::
    \\u_t=actGate(xu_t+W_{u}h_{t-1}+b_u)\\
    \\r_t=actGate(xr_t+W_{r}h_{t-1}+b_r)\\
    \\m_t=actNode(xm_t+W_{c}dot(r_t,h_{t-1})+b_m)\\
    \\h_t=dot((1-u_t),m_t)+dot(u_t,h_{t-1})\\
    
 
 
GRU单元的输入包括 :math:`z_t` ， :math:`h_{t-1}` 。在上述等式中， :math:`z_t` 会被分割成三部分： :math:`xu_t` 、 :math:`xr_t` 和 :math:`xm_t`  。
这意味着要为一批输入实现一个全连接层，我们需要采用一个全连接层，才能得到 :math:`z_t=W_{fc}x_t` 。
:math:`u_t` 和 :math:`r_t` 分别代表了GRU神经元的update gates（更新门）和reset gates(重置门)。
和LSTM不同，GRU少了一个门（它没有LSTM的forget gate）。但是它有一个叫做中间候选隐藏状态（intermediate candidate hidden output）的输出，
记为 :math:`m_t` 。 该层有三个输出： :math:`h_t, dot(r_t,h_{t-1})` 以及 :math:`u_t，r_t，m_t` 的连结(concatenation)。
 
 


参数:
  - **input** (Variable) – 经FC层变换后的当前步骤的输入值
  - **hidden** (Variable) –  从上一步而来的gru unit 隐藏状态值(hidden value)
  - **size** (integer) – 输入数据的维度
  - **param_attr** (ParamAttr|None) – 可学习的隐藏层权重矩阵的参数属性。
    注意：
      - 该权重矩阵形为 :math:`(T×3D)` ，D是隐藏状态的规模（hidden size）
      - 该权重矩阵的所有元素由两部分组成， 一是update gate和reset gate的权重，形为 :math:`(D×2D)` ；二是候选隐藏状态（candidate hidden state）的权重矩阵，形为 :math:`(D×D)`
    如果该函数参数值为None或者 ``ParamAttr`` 类中的属性之一，gru_unit则会创建一个 ``ParamAttr`` 类的对象作为 param_attr。如果param_attr没有被初始化，那么会由Xavier来初始化它。默认值为None
  - **bias_attr** (ParamAttr|bool|None) - GRU的bias变量的参数属性。本bias会连结（concatenate）在update gates（更新门），reset gates(重置门)以及candidate calculations（候选隐藏状态计算）中的bias。如果值为False，那么上述三者将没有bias参与运算。若值为None或者 ``ParamAttr`` 类中的属性之一，gru_unit则会创建一个 ``ParamAttr`` 类的对象作为 bias_attr。如果bias_attr没有被初始化，那它会被默认初始化为0。默认值为None。
  - **activation** (string) –  神经元 “actNode” 的激励函数（activation）类型。默认类型为‘tanh’
  - **gate_activation** (string) – 门 “actGate” 的激励函数（activation）类型。 默认类型为 ‘sigmoid’
  

返回：	 hidden value（隐藏状态的值），reset-hidden value(重置隐藏状态值)，gate values(门值)

返回类型:	 元组（tuple）


**代码示例**

..  code-block:: python

    # 假设我们现在有x_t_data和size=10的之前的隐藏层
    x_t = fluid.layers.fc(input=x_t_data, size=30)
    hidden_val, r_h_val, gate_val = fluid.layers.gru_unit(input=x_t,
                                          hidden = prev_hidden)





