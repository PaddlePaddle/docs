.. _cn_api_fluid_layers_dynamic_lstmp:

dynamic_lstmp
-------------------------------
.. py:function:: paddle.fluid.layers.dynamic_lstmp(input, size, proj_size, param_attr=None, bias_attr=None, use_peepholes=True, is_reverse=False, gate_activation='sigmoid', cell_activation='tanh', candidate_activation='tanh', proj_activation='tanh', dtype='float32', name=None, h_0=None, c_0=None, cell_clip=None, proj_clip=None)

动态LSTMP层(Dynamic LSTMP Layer)

LSTMP层(具有循环映射的LSTM)在LSTM层后有一个分离的映射层，从原始隐藏状态映射到较低维的状态，用来减少参数总数，减少LSTM计算复杂度，特别是输出单元相对较大的情况下。 `Long Short-Term Memory Recurrent Neural Network Architectures for Large Scale Acoustic Modeling <https://research.google.com/pubs/archive/43905.pdf>`_

公式如下：

.. math::

        i_t & = \sigma(W_{ix}x_{t} + W_{ir}r_{t-1} + W_{ic}c_{t-1} + b_i)\\
        f_t & = \sigma(W_{fx}x_{t} + W_{fr}r_{t-1} + W_{fc}c_{t-1} + b_f)\\
        \tilde{c_t} & = act_g(W_{cx}x_t + W_{cr}r_{t-1} + b_c)\\
        o_t & = \sigma(W_{ox}x_{t} + W_{or}r_{t-1} + W_{oc}c_t + b_o)\\
        c_t & = f_t \odot c_{t-1} + i_t \odot \tilde{c_t}\\
        h_t & = o_t \odot act_h(c_t)\\
        r_t & = \overline{act_h}(W_{rh}h_t)\\


在以上公式中：
    - :math:`W` : 代表权重矩阵（例如 :math:`W_{xi}` 是输入门道输入的权重矩阵）
    - :math:`W_{ic}` , :math:`W_{fc}` , :math:`W_{oc}`  : peephole connections的对角权重矩阵。在我们的实现中，外面用向量代表这些对角权重矩阵
    - :math:`b` : 代表偏差向量（例如 :math:`b_{i}` 是输入偏差向量）
    - :math:`\delta` : 激活函数，比如逻辑回归函数
    - :math:`i,f,o` 和 :math:`c` :分别代表输入门，遗忘门,输出门和cell激活函数向量，四者的大小和cell输出激活函数向量 :math:`h` 的四者大小相等
    - :math:`h` : 隐藏状态
    - :math:`r` : 隐藏状态的循环映射
    - :math:`\tilde{c_t}` : 候选隐藏状态
    - :math:`\odot` : 向量的元素状态生成
    - :math:`act_g` 和 :math:`act_h` : cell输入和cell输出激活函数，通常使用 :math:`tanh`
    - :math:`\overline{act_h}` : 映射输出的激活函数，通常用 :math:`identity` 或等同的 :math:`act_h`

将 ``use_peepholes`` 设置为False，断开窥视孔连接（peephole connection）。在此省略公式，详情请参照论文 `LONG SHORT-TERM MEMORY <http://www.bioinf.jku.at/publications/older/2604.pdf>`_ 。

注意输入 :math:`x_{t}` 中的 :math:`W_{xi}x_{t},W_{xf}x_{t},W_{xc}x_{t},W_{xo}x_{t}` 不在此操作符中。用户选择在LSTMP层之前使用全链接层。

参数：
    - **input** (Variable) - dynamic_lstmp层的输入，支持输入序列长度为变量的倍数。该变量的张量为一个矩阵，维度为（T X 4D），T为mini-batch的总时间步长，D是隐藏大小。
    - **size** (int) - 4*隐藏状态大小（hidden size）
    - **proj_size** (int) - 投影输出的大小
    - **param_attr** (ParamAttr|None) -   可学习hidden-hidden权重和投影权重的参数属性。
      说明:
        - Hidden-hidden （隐藏状态到隐藏状态）权重 = :math:`\{ W_{ch},W_{ih},W_{fh},W_{oh} \}`
        - hidden-hidden权重的权重矩阵为（P*4D），P是投影大小，D是隐藏大小。
        - 投影（Projection）权重 = :math:`\{ W_{rh} \}`
        - 投影权重的shape为（D\*P）

      如果设为None或者ParamAttr的一个属性，dynamic_lstm将创建ParamAttr为param_attr。如果param_attr的初始函数未设置，参数则初始化为Xavier。默认:None。
    - **bias_attr** (ParamAttr|None) - 可学习bias权重的bias属性，包含输入隐藏的bias权重和窥视孔连接权重（peephole connection）,前提是use_peepholes设为True。

      说明:
        1.use_peepholes = False
            - Biases = { :math:`b_{c},b_{i},b_{f},b_{o}`}.
            - 维度为（1*4D）

        2.use_peepholes = True
            - Biases = { :math:`b_{c},b_{i},b_{f},b_{o},W_{ic},W_{fc},W_{oc}`}
            - 维度为（1*7D）

        如果设置为None或者ParamAttr的一个属性，dynamic_lstm将创建ParamAttr为bias_attr。bias_attr的初始函数未设置，bias则初始化为0.默认：None。

    - **use_peepholes** (bool) - 是否开启诊断/窥视孔链接，默认为True。
    - **is_reverse** (bool) - 是否计算反向LSTM，默认为False。
    - **gate_activation** (bool) - 输入门（input gate）、遗忘门（forget gate）和输出门（output gate）的激活函数。Choices = [“sigmoid”，“tanh”，“relu”，“identity”]，默认“sigmoid”。
    - **cell_activation** (str) - cell输出的激活函数。Choices = [“sigmoid”，“tanh”，“relu”，“identity”]，默认“tanh”。
    - **candidate_activation** (str) - 候选隐藏状态（candidate hidden state）的激活状态。Choices = [“sigmoid”，“tanh”，“relu”，“identity”]，默认“tanh”。
    - **proj_activation** (str) - 投影输出的激活函数。Choices = [“sigmoid”，“tanh”，“relu”，“identity”]，默认“tanh”。
    - **dtype** (str) - 数据类型。Choices = [“float32”，“float64”]，默认“float32”。
    - **name** (str|None) - 该层名称（可选）。若设为None，则自动为该层命名。
    - **h_0** (Variable) - 初始隐藏状态是可选输入，默认为0。这是一个具有形状的张量(N x D)，其中N是批大小，D是投影大小。
    - **c_0** (Variable) - 初始cell状态是可选输入，默认为0。这是一个具有形状(N x D)的张量，其中N是批大小。h_0和c_0可以为空，但只能同时为空。
    - **cell_clip** (float) - 如果提供该参数，则在单元输出激活之前，单元状态将被此值剪裁。
    - **proj_clip** (float) - 如果 num_proj > 0 并且 proj_clip 被提供,那么将投影值沿元素方向剪切到[-proj_clip，proj_clip]内

返回：含有两个输出变量的元组，隐藏状态（hidden state）的投影和LSTMP的cell状态。投影的shape为（T*P），cell state的shape为（T*D），两者的LoD和输入相同。

返回类型：元组(tuple)

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    dict_dim, emb_dim = 128, 64
    data = fluid.layers.data(name='sequence', shape=[1],
                         dtype='int32', lod_level=1)
    emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])
    hidden_dim, proj_dim = 512, 256
    fc_out = fluid.layers.fc(input=emb, size=hidden_dim * 4,
                         act=None, bias_attr=None)
    proj_out, _ = fluid.layers.dynamic_lstmp(input=fc_out,
                                         size=hidden_dim * 4,
                                         proj_size=proj_dim,
                                         use_peepholes=False,
                                         is_reverse=True,
                                         cell_activation="tanh",
                                         proj_activation="tanh")











