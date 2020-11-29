.. _cn_user_guide_broadcasting:

==================
广播 (broadcasting)
==================

飞桨（PaddlePaddle，以下简称Paddle）和其他框架一样，提供的一些API支持广播(broadcasting)机制，允许在一些运算时使用不同形状的张量。
通常来讲，如果有一个形状较小和一个形状较大的张量，我们希望多次使用较小的张量来对较大的张量执行一些操作，看起来像是较小形状的张量的形状首先被扩展到和较大形状的张量一致，然后做运算。
值得注意的是，这期间并没有对较小形状张量的数据拷贝操作。

飞桨的广播机制主要遵循如下规则；如果两个张量的形状遵循一下规则，我们认为这两个张量是可广播的（参考`Numpy 广播机制 <https://numpy.org/doc/stable/user/basics.broadcasting.html#module-numpy.doc.broadcasting>`）：

1. 每个张量至少为一维张量
2. 从后往前比较张量的形状，当前维度的大小要么相等，要么其中一个等于一，要么其中一个不存在

例如：

.. code-block:: python

    import paddle
    import numpy as np

    paddle.enable_imperative()

    x = paddle.imperative.to_variable(np.ones((2,3,4), np.float32))
    y = paddle.imperative.to_variable(np.ones((2,3,4), np.float32))
    # 两个张量 形状一致，可以广播

    x = paddle.imperative.to_variable(np.ones((2,3,1,5), np.float32))
    y = paddle.imperative.to_variable(np.ones((3,4,1), np.float32))
    # 从后向前依次比较：
    # 第一次：y的维度大小是1
    # 第二次：x的维度大小是1
    # 第三次：x和y的维度大小相等
    # 第四次：y的维度不存在
    # 所以 x和y是可以广播的

    # 相反
    x = paddle.imperative.to_variable(np.ones((2,3,4), np.float32))
    y = paddle.imperative.to_variable(np.ones((2,3,6), np.float32))
    # 此时x和y是不可广播的，因为第一次比较 4不等于6

现在我们知道什么情况下两个张量是可以广播的，两个张量进行广播语义后的结果张量的形状计算规则如下：

1. 如果两个张量的形状的长度不一致，那么需要在较小形状长度的矩阵像前添加1，只到两个张量的形状长度相等。
2. 保证两个张量形状相等之后，每个维度上的结果维度就是当前维度上较大的那个。

例如:

.. code-block:: python

    import paddle
    import numpy as np

    paddle.enable_imperative()

    x = paddle.imperative.to_variable(np.ones((2,1,4), np.float32))
    y = paddle.imperative.to_variable(np.ones((3,1), np.float32))
    z = x+y
    print(z.shape)
    # z的形状: [2,3,4]

    x = paddle.imperative.to_variable(np.ones((2,1,4), np.float32))
    y = paddle.imperative.to_variable(np.ones((3,2), np.float32))
    z = x+y
    print(z.shape)
    # InvalidArgumentError: Broadcast dimension mismatch.

除此之外，飞桨的elementwise系列API针对广播机制增加了axis参数，当使用较小形状的y来来匹配较大形状的x的时候，且满足y的形状的长度小于x的形状长度，
axis表示y在x上应用广播机制的时候的起始维度的位置，当设置了asis参数后，张量的维度比较顺序变成了从axis开始，从前向后比较。当axis=-1时，axis = rank(x) - rank(y),
同时y的大小为1的尾部维度将被忽略。

例如:

.. code-block:: python

    import paddle
    import numpy as np

    paddle.enable_imperative()

    x = paddle.imperative.to_variable(np.ones((2,1,4), np.float32))
    y = paddle.imperative.to_variable(np.ones((3,1), np.float32))
    z = paddle.elementwise_add(x,y,axis=1)
    # z的形状 [2, 3, 4]

    x = paddle.imperative.to_variable(np.ones((2,3,4,5), np.float32))
    y = paddle.imperative.to_variable(np.ones((4,5), np.float32))
    z = paddle.elementwise_add(x,y,axis=1)
    print(z.shape)
    # InvalidArgumentError: Broadcast dimension mismatch.
    # 因为指定了axis之后，计算广播的维度从axis开始从前向后比较

    x = paddle.imperative.to_variable(np.ones((2,3,4,5), np.float32))
    y = paddle.imperative.to_variable(np.ones((3), np.float32))
    z = paddle.elementwise_add(x,y,axis=1)
    print(z.shape)
    # z的形状 [2, 3, 4, 5]
    # 因为此时是从axis=1的维度开始，从前向后比较维度进行广播
