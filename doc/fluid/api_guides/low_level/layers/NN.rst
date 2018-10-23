..  _api_guide_NN:

########
神经网络
########

神经网络的发展加深了人工智能领域的知识厚度，也促进了深度学习近年来的蓬勃发展。
PaddlePaddle Fluid 将NN的相关api存放在 layers 下的NN中。

下面介绍PaddlePaddle Fluid NN相关的Api。


---------------

:code:`全连接层`：创建全连接层，接受多个张量作为输入，并为每个输入张量创建一个权重，相关API Reference 请参考 fc_ 。

.. _fc: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#permalink-30-fc

:code:`LSTM`：LSTM 和 LSTMP OP的相关API Reference请参考 LSTM_ 和 LSTMP_ 。

.. _LSTM: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#permalink-32-dynamic_lstm
.. _LSTMP: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#permalink-33-dynamic_lstmpy

:code:`GRU layer`：相关API Reference 请参考 dynamic_gru_ 和 gru_unit_ 。

.. _dynamic_gru: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#permalink-34-dynamic_gru

.. _gru_unit: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#permalink-35-gru_unit

:code:`线性链条件随机场`： 相关API Reference 请参考 linear_chain_crf_ 和 crf_decoding_ 。

.. _linear_chain_crf: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#permalink-36-linear_chain_crf
.. _crf_decoding: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#permalink-37-crf_decoding

:code:`softmax`：softmax OP，相关API Reference 请参考 softmax_ 。

.. _softmax: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#permalink-47-softmax

:code:`batch_norm`：接受NHWC或NCHW形式的输入数据。 相关API Reference 请参考 batch_norm_ 。

.. _barch_norm: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#permalink-50-batch_norm

:code:`基础运算`：计算张量的平均值，和，最大值，最小值，乘积，dim参数指定计算实施的维度。相关API Reference 请参考
sum_ ， mean_ ， max_ ， product_ 。

.. _sum: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#permalink-58-reduce_sum
.. _mean: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#permalink-59-reduce_mean
.. _max: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#permalink-60-reduce_max
.. _min: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#permalink-61-reduce_min
.. _product: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#permalink-62-reduce_prod

:code:`dropout`：计算dropout，删除或者保留X的每个元素，dropout是正则化技术，在训练过程中用来削弱神经元间的互相适应从而避免过拟合问题，
dropout随机的将一些单元的输出设置为0。相关API Reference 请参考 dropout_ 。

.. _dropout: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#permalink-65-dropout

:code:`matmul`：计算张量的乘法， 相关API Reference 请参考 matmul_ 。

.. _matmul: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#permalink-70-matmul

:code:`split`：split方法将一个张量分成子张量， 相关API Reference 请参考 split_ 。

.. _split: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#permalink-66-split

:code:`topk`：如名字所示，topk返回topk的值和下标， 相关API Reference 请参考 topk_ 。

.. _topk: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#permalink-71-topk

:code:`L2正则化`：L2正则化通过添加正则化项致力于减少参数平方的综合， 相关API Reference 请参考 l2_normalize_ 。

.. _l2_normalize : http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#permalink-69-l2_normalize

:code:`smooth_l1`：用于计算变量x和y的smooth L1 损失， 相关API Reference 请参考 smooth_l1_ 。

.. _smooth_l1: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#permalink-83-smooth_l1

:code:`one_hot`：one_hot编码， 相关API Reference 请参考 one.hot_ 。

.. _one.hot: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#permalink-84-one_hot

:code:`reshape`：改变张量的维数，其中有如下的技巧，将待reshape的张量的一个维度设置为-1，
代表着这个维度的值将由x的总元素数量和剩余维度推断而来，显然，有且只有一个维度能设置为-1。相关API Reference 请参考 reshape_ 。

.. _reshape: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#permalink-86-reshape

:code:`squeeze`：从张量中移除单维度条目，即把shape中为1的维度去掉，axes用于指定需要删除的维度，若axes为空，则删除所有单维度的条目，
相关API Reference 请参考 sequeeze_ 。

.. _sequeeze: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#permalink-87-squeeze

:code:`LRN`：局部响应归一化，在AlexNet中提出了LRN层，对局部神经元的活动创建竞争机制，
使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，增强了模型的泛化能力， 相关API Reference 请参考 lrn_ 。

.. _lrn: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#permalink-90-lrn

:code:`pad`：fluid通过pad对张量进行填充，相关API Reference 请参考 pad_ 和 pad_constant_like_ 。

.. _pad: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#permalink-91-pad
.. _pad_constant_like: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#permalink-92-pad_constant_like

:code:`LSR`：是一种通过在输出中添加噪声，从而降低模型过拟合的一种方法， 相关API Reference 请参考 label_smooth_  。

.. _label_smooth: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#permalink-93-label_smooth

:code:`roi_pooling`: 相关API Reference 请参考 roi_pool_ 。

.. _roi_pool: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#permalink-94-roi_pool

:code:`image_resize`：按宽高对图片进行缩放，相关API Reference 请参考 image_resize_ 和 image_resize_short_ 。

.. _image_resize: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#permalink-96-image_resize
.. _image_resize_short: http ://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#permalink-97-image_resize_short

:code:`gather`:按照index给出的值对集合进行抽取，适合抽取不连续区域的子集，相关API Reference 请参考 gather_ 。

.. _gather: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#permalink-99-gather
