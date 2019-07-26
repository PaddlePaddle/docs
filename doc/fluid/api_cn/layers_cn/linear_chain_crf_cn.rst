.. _cn_api_fluid_layers_linear_chain_crf:

linear_chain_crf
-------------------------------

.. py:function:: paddle.fluid.layers.linear_chain_crf(input, label, param_attr=None)

线性链条件随机场（Linear Chain CRF）

条件随机场定义间接概率图，节点代表随机变量，边代表两个变量之间的依赖。CRF学习条件概率 :math:`P\left ( Y|X \right )` ， :math:`X = \left ( x_{1},x_{2},...,x_{n} \right )` 是结构性输入，:math:`Y = \left ( y_{1},y_{2},...,y_{n} \right )` 为输入标签。

线性链条件随机场（Linear Chain CRF)是特殊的条件随机场（CRF），有利于序列标注任务。序列标注任务不为输入设定许多条件依赖。唯一的限制是输入和输出必须是线性序列。因此类似CRF的图是一个简单的链或者线，也就是线性链随机场（linear chain CRF）。

该操作符实现了线性链条件随机场（linear chain CRF）的前向——反向算法。详情请参照 http://www.cs.columbia.edu/~mcollins/fb.pdf 和 http://cseweb.ucsd.edu/~elkan/250Bwinter2012/loglinearCRFs.pdf。


长度为L的序列s的概率定义如下：

.. math::

    P(s) = (1/Z) exp(a_{s_1} + b_{s_L} + sum_{l=1}^L x_{s_l} + sum_{l=2}^L w_{s_{l-1},s_l})


其中Z是正则化值，所有可能序列的P(s)之和为1，x是线性链条件随机场（linear chain CRF）的发射（emission）特征权重。

线性链条件随机场最终输出mini-batch每个训练样本的条件概率的对数


  1.这里 :math:`x` 代表Emission

  2.Transition的第一维度值，代表起始权重，这里用 :math:`a` 表示

  3.Transition的下一维值，代表末尾权重，这里用 :math:`b` 表示

  4.Transition剩下的值，代表转移权重，这里用 :math:`w` 表示

  5.Label用 :math:`s` 表示




**注意：**

    1.条件随机场（CRF）的特征函数由发射特征(emission feature）和转移特征（transition feature）组成。发射特征（emission feature）权重在调用函数前计算，而不在函数里计算。

    2.由于该函数对所有可能序列的进行全局正则化，发射特征（emission feature）权重应是未缩放的。因此如果该函数带有发射特征（emission feature），并且发射特征是任意非线性激活函数的输出，则请勿调用该函数。

    3.Emission的第二维度必须和标记数字（tag number）相同

参数：
    - **input** (Variable，LoDTensor，默认float类型LoDTensor) - 一个二维LoDTensor，shape为[N*D]，N是mini-batch的大小，D是总标记数。线性链条件随机场的未缩放发射权重矩阵
    - **input** (Tensor，默认float类型LoDTensor) - 一个二维张量，shape为[(D+2)*D]。linear_chain_crf操作符的可学习参数。更多详情见operator注释
    - **label** (Variable，LoDTensor，默认int64类型LoDTensor） - shape为[N*10的LoDTensor，N是mini-batch的总元素数
    - **param_attr** (ParamAttr) - 可学习参数的属性

返回：
    output(Variable，Tensor，默认float类型Tensor)：shape为[N*D]的二维张量。Emission的指数。这是前向计算中的中间计算结果，在后向计算中还会复用

    output(Variable，Tensor，默认float类型Tensor)：shape为[(D+2)*D]的二维张量。Transition的指数。这是前向计算中的中间计算结果，在后向计算中还会复用

    output(Variable,Tensor，默认float类型Tensor)：mini-batch每个训练样本的条件概率的对数。这是一个shape为[S*1]的二维张量，S是mini-batch的序列数。注：S等同于mini-batch的序列数。输出不再是LoDTensor

返回类型：output（Variable）

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    emission = fluid.layers.data(name='emission', shape=[1000], dtype='float32')
    target = fluid.layers.data(name='target', shape=[1], dtype='int32')
    crf_cost = fluid.layers.linear_chain_crf(
        input=emission,
        label=target,
        param_attr=fluid.ParamAttr(
            name='crfw',
            learning_rate=0.2))











