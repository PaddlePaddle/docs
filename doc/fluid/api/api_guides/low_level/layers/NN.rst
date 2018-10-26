..  _api_guide_NN:

########
神经网络
########

神经网络的发展加深了人工智能领域的知识厚度，也促进了深度学习近年来的蓬勃发展。
PaddlePaddle Fluid 将神经网络（NN）的相关api存放在 layers 下的NN中。

下面介绍PaddlePaddle Fluid NN相关的Api。

NN相关
---------------

:code:`fc`：即Fully Connected Layer，全连接层。创建全连接层，接受多个张量作为输入，并为每个输入张量创建一个权重。
相关API Reference 请参考 :ref:`api_fluid_layers_fc`

:code:`embedding`：即嵌入层。
相关API Reference 请参考 :ref:`api_fluid_layers_embedding`

:code:`one_hot`：即one_hot编码
相关API Reference 请参考 :ref:`api_fluid_layers_one_hot`

:code:`LSTM`：即Long Short-Term Memory，长短期记忆网络。
LSTM和LSTMP OP的相关API Reference请参考 :ref:`api_fluid_layers_dynamic_lstm` 和 :ref:`api_fluid_layers_dynamic_lstmp`

:code:`GRU`：即gated recurrent unit，门控循环单元。
相关API Reference 请参考 :ref:`api_fluid_layers_dynamic_gru` 和  :ref:`api_fluid_layers_gru_unit`

:code:`CRF`：即Conditional Random Fields，条件随机场。
相关API Reference 请参考 :ref:`api_fluid_layers_linear_chain_crf` 和 :ref:`api_fluid_layers_crf_decoding`

:code:`softmax`：即softmax OP。
相关API Reference 请参考  :ref:`api_fluid_layers_softmax`


正则化与归一化
---------------

:code:`LRN`：即局部响应归一化，在AlexNet中提出了LRN层，对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，增强了模型的泛化能力。
相关API Reference 请参考 :ref:`api_fluid_layers_lrn`

:code:`batch_norm`：即批标准化，接受NHWC或NCHW形式的输入数据。
相关API Reference 请参考 :ref:`api_fluid_layers_batch_norm`

:code:`dropout`：计算dropout，删除或者保留X的每个元素，dropout是正则化技术，在训练过程中用来削弱神经元间的互相适应从而避免过拟合问题，dropout随机的将一些单元的输出设置为0。
相关API Reference 请参考 :ref:`api_fluid_layers_dropout`

:code:`smooth_l1`：用于计算变量x和y的smooth L1 损失。
相关API Reference 请参考 :ref:`api_fluid_layers_smooth_l1`

:code:`L2正则化`：L2正则化通过添加正则化项致力于减少参数平方的综合。
相关API Reference 请参考 :ref:`api_fluid_layers_l2_normalize`

:code:`LSR`：是一种通过在输出中添加噪声，从而降低模型过拟合的一种方法。
相关API Reference 请参考 :ref:`api_fluid_layers_label_smooth`

:code:`nce`：即Noise Contrastive Estimation，噪声对比估计。
相关API Reference 请参考 :ref:`api_fluid_layers_nce`


基础运算与变换
---------------

:code:`matmul`：计算张量的乘法。
相关API Reference 请参考 :ref:`api_fluid_layers_matmul`

:code:`split`：split方法将一个张量分成子张量。
相关API Reference 请参考 :ref:`api_fluid_layers_split`

:code:`topk`：返回top k的值和下标。
相关API Reference 请参考 :ref:`api_fluid_layers_topk`

:code:`transpose`：转置张量。
相关API Reference 请参考 :ref:`api_fluid_layers_transpose`

:code:`reshape`：改变张量的维数。其中，将待reshape的张量的一个维度设置为-1，代表着这个维度的值将由x的总元素数量和剩余维度推断而来，显然，有且只有一个维度能设置为-1。
相关API Reference 请参考 :ref:`api_fluid_layers_reshape`

:code:`squeeze`：从张量中移除单维度条目，即把shape中为1的维度去掉，axes用于指定需要删除的维度，若axes为空，则删除所有单维度的条目，
相关API Reference 请参考 :ref:`api_fluid_layers_squeeze`

:code:`pad`：fluid通过pad对张量进行填充。
相关API Reference 请参考 :ref:`api_fluid_layers_pad` 和 :ref:`api_fluid_layers_pad_constant_like`

:code:`image_resize`：按宽高对图片进行缩放。
相关API Reference 请参考 :ref:`api_fluid_layers_label_image_resize` 和 :ref:`api_fluid_layers_label_image_resize_short`

:code:`gather`:按照index给出的值对集合进行抽取，适合抽取不连续区域的子集。
相关API Reference 请参考 :ref:`api_fluid_layers_gather`
