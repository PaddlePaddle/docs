.. _cn_api_paddle_nn_TripletMarginLoss:

TripletMarginLoss
-------------------------------

.. py:class:: paddle.nn.TripletMarginLoss(margin: float = 1.0, p: float = 2., epsilon: float = 1e-6, swap: bool = False,reduction: str = 'mean', name:str=None)

创建一个 TripletMarginLoss 的可调用类。通过计算输入 `input` 和 `positive` 和 `negative` 间的 `triplet margin loss` 损失，测量样本之间，即 `input` 与 `positive examples` 和 `negative examples` 的相对相似性。

损失函数按照下列公式计算

.. math::
    L(input, pos, neg) = \max \{d(input_i, pos_i) - d(input_i, neg_i) + {\rm margin}, 0\}


其中的

.. math::
    d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_p


``p`` 为距离函数的范数。``margin`` 为（input，positive）与（input，negative）的距离间隔，``swap`` 为 True 时，会比较（input，negative）和（positive，negative）的大小，并将（input，negative）换为其中较小的值，内容详见论文 `Learning shallow convolutional feature descriptors with triplet losses <http://www.bmva.org/bmvc/2016/papers/paper119/paper119.pdf>`_ 。

参数
:::::::::
    - **margin** (float，可选) - 手动指定间距，默认为 1。
    - **p** (float，可选) - 手动指定范数，默认为 2。
    - **epsilon** (float，可选) - 防止除数为 0，默认为 1e-6。
    - **swap** (bool，可选) - 默认为 False。
    - **reduction** (str，可选) - 指定应用于输出结果的计算方式，可选值有：``'none'``、``'mean'``、``'sum'``。默认为 ``'mean'``，计算 Loss 的均值；设置为 ``'sum'`` 时，计算 Loss 的总和；设置为 ``'none'`` 时，则返回原始 Loss。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
:::::::::
    - **input** (Tensor) - :math:`[N, *]`，其中 N 是 batch_size， `*` 是任意其他维度。数据类型是 float32、float64。
    - **positive** (Tensor) - :math:`[N, *]`，标签 ``positive`` 的维度、数据类型与输入 ``input`` 相同。
    - **negative** (Tensor) - :math:`[N, *]`，标签 ``negative`` 的维度、数据类型与输入 ``input`` 相同。
    - **output** (Tensor) - 输出的 Tensor。如果 :attr:`reduction` 是 ``'none'``，则输出的维度为 :math:`[N, *]`，与输入 ``input`` 的形状相同。如果 :attr:`reduction` 是 ``'mean'`` 或 ``'sum'``，则输出的维度为 :math:`[1]` 。

返回
:::::::::
   返回计算 TripletMarginLoss 的可调用对象。

代码示例
:::::::::
COPY-FROM: paddle.nn.TripletMarginLoss
