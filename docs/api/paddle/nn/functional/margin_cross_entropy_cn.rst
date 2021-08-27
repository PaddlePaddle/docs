.. _cn_api_paddle_nn_functional_margin_cross_entropy:

margin_cross_entropy
-------------------------------

.. py:function:: paddle.nn.functional.margin_cross_entropy(logits, label, margin1=1.0, margin2=0.5, margin3=0.0, scale=64.0, group=None, return_softmax=False, reduction='mean')

.. math::
    L=-\\frac{1}{N}\sum^N_{i=1}\log\\frac{e^{s(cos(m_{1}\\theta_{y_i}+m_{2})-m_{3})}}{e^{s(cos(m_{1}\\theta_{y_i}+m_{2})-m_{3})}+\sum^n_{j=1,j\\neq y_i} e^{scos\\theta_{y_i}}}

其中，:math:`\\theta_{y_i}` 是特征 :math:`x` 与类 :math:`i` 的角度。更详细的介绍请参考 ``Arcface loss``，https://arxiv.org/abs/1801.07698 。

提示:
    这个 API 支持单卡，也支持多卡（模型并行），使用模型并行，``logits.shape[-1]`` 在每张卡上可以不同。

参数:
    - **logits** (Tensor) - 2-D Tensor，维度为 ``[N, local_num_classes]``，``logits`` 为归一化后的 ``X`` 与归一化后的 ``W`` 矩阵乘得到，数据类型为 float16，float32 或者 float64。如果用了模型并行，则``logits == sahrd_logits``。
    - **label** (Tensor) - 维度为 ``[N]`` 或者 ``[N, 1]`` 的标签。
    - **margin1** (float，可选) - 公式中的 ``m1``。默认值为 ``1.0``。
    - **margin2** (float，可选) - 公式中的 ``m2``。默认值为 ``0.5``。
    - **margin3** (float，可选) - 公式中的 ``m3``。默认值为 ``0.0``。
    - **scale** (float，可选) - 公式中的 ``s``。默认值为 ``64.0``。
    - **group** (Group, 可选) - 通信组的抽象描述，具体可以参考 ``paddle.distributed.collective.Group``。默认值为 ``None``。
    - **return_softmax** (bool，可选) - 是否返回 ``softmax`` 概率值。默认值为 ``None``。
    - **reduction** （str, 可选）- 是否对 ``loss`` 进行归约。可选值为 ``'none'`` | ``'mean'`` | ``'sum'``。如果 ``reduction='mean'``，则对 ``loss`` 进行平均，如果``reduction='sum'``，则对 ``loss`` 进行求和，``reduction='None'``，则直接返回 ``loss``。默认值为 ``'mean'``。

返回:
    - ``Tensor`` (``loss``) 或者 ``Tensor`` 二元组 (``loss``, ``softmax``) - 如果 ``return_softmax=False`` 返回 ``loss``，否则返回 (``loss``, ``softmax``)。当使用模型并行时 ``softmax == shard_softmax``，否则 ``softmax`` 的维度与 ``logits`` 相同。如果 ``reduction == None``，``loss`` 的维度为 ``[N, 1]``，否则为 ``[1]``。

抛出异常:
    - :code:`ValueError` - ``logits_dims - 1 != label_dims and logits_dims != label_dims`` 时抛出异常。

**代码示例**:

.. code-block:: python

    # required: gpu
    # Single GPU
    import paddle
    m1 = 1.0
    m2 = 0.5
    m3 = 0.0
    s = 64.0
    batch_size = 2
    feature_length = 4
    num_classes = 4
    label = paddle.randint(low=0, high=num_classes, shape=[batch_size], dtype='int64')
    X = paddle.randn(
        shape=[batch_size, feature_length],
        dtype='float64')
    X_l2 = paddle.sqrt(paddle.sum(paddle.square(X), axis=1, keepdim=True))
    X = paddle.divide(X, X_l2)
    W = paddle.randn(
        shape=[feature_length, num_classes],
        dtype='float64')
    W_l2 = paddle.sqrt(paddle.sum(paddle.square(W), axis=0, keepdim=True))
    W = paddle.divide(W, W_l2)
    logits = paddle.matmul(X, W)
    loss, softmax = paddle.nn.functional.margin_cross_entropy(
        logits, label, margin1=m1, margin2=m2, margin3=m3, scale=s, return_softmax=True, reduction=None)
    print(logits)
    print(label)
    print(loss)
    print(softmax)

    #Tensor(shape=[2, 4], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
    #       [[ 0.85204151, -0.55557678,  0.04994566,  0.71986042],
    #        [-0.20198586, -0.35270476, -0.55182702,  0.09749021]])
    #Tensor(shape=[2], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
    #       [2, 3])
    #Tensor(shape=[2, 1], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
    #       [[82.37059586],
    #        [12.13448420]])
    #Tensor(shape=[2, 4], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
    #       [[0.99978819, 0.00000000, 0.00000000, 0.00021181],
    #        [0.99992995, 0.00006468, 0.00000000, 0.00000537]])
    
.. code-block:: python

    # required: distributed
    # Multi GPU, test_margin_cross_entropy.py
    import paddle
    import paddle.distributed as dist
    strategy = dist.fleet.DistributedStrategy()
    dist.fleet.init(is_collective=True, strategy=strategy)
    rank_id = dist.get_rank()
    m1 = 1.0
    m2 = 0.5
    m3 = 0.0
    s = 64.0
    batch_size = 2
    feature_length = 4
    num_class_per_card = [4, 8]
    num_classes = paddle.sum(paddle.to_tensor(num_class_per_card))
    label = paddle.randint(low=0, high=num_classes.item(), shape=[batch_size], dtype='int64')
    label_list = []
    dist.all_gather(label_list, label)
    label = paddle.concat(label_list, axis=0)
    X = paddle.randn(
        shape=[batch_size, feature_length],
        dtype='float64')
    X_list = []
    dist.all_gather(X_list, X)
    X = paddle.concat(X_list, axis=0)
    X_l2 = paddle.sqrt(paddle.sum(paddle.square(X), axis=1, keepdim=True))
    X = paddle.divide(X, X_l2)
    W = paddle.randn(
        shape=[feature_length, num_class_per_card[rank_id]],
        dtype='float64')
    W_l2 = paddle.sqrt(paddle.sum(paddle.square(W), axis=0, keepdim=True))
    W = paddle.divide(W, W_l2)
    logits = paddle.matmul(X, W)
    loss, softmax = paddle.nn.functional.margin_cross_entropy(
        logits, label, margin1=m1, margin2=m2, margin3=m3, scale=s, return_softmax=True, reduction=None)
    print(logits)
    print(label)
    print(loss)
    print(softmax)
    # python -m paddle.distributed.launch --gpus=0,1 test_margin_cross_entropy.py 
    ## for rank0 input
    #Tensor(shape=[4, 4], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
    #       [[ 0.32888934,  0.02408748, -0.02763289,  0.18173063],
    #        [-0.52893978, -0.10623845, -0.21596515, -0.06432517],
    #        [-0.00536345, -0.03924667,  0.66735314, -0.28640926],
    #        [-0.09907366, -0.48534973, -0.10365338, -0.39472322]])
    #Tensor(shape=[4], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
    #       [11, 1 , 10, 11])
    ## for rank1 input
    #Tensor(shape=[4, 8], dtype=float64, place=CUDAPlace(1), stop_gradient=True,
    #       [[ 0.68654754,  0.28137170,  0.69694954, -0.60923933, -0.57077653,  0.54576703, -0.38709028,  0.56028204],
    #        [-0.80360371, -0.03042448, -0.45107338,  0.49559349,  0.69998950, -0.45411693,  0.61927630, -0.82808600],
    #        [ 0.11457570, -0.34785879, -0.68819499, -0.26189226, -0.48241491, -0.67685711,  0.06510185,  0.49660849],
    #        [ 0.31604851,  0.52087884,  0.53124749, -0.86176582, -0.43426329,  0.34786144, -0.10850784,  0.51566383]])
    #Tensor(shape=[4], dtype=int64, place=CUDAPlace(1), stop_gradient=True,
    #       [11, 1 , 10, 11])
    ## for rank0 output
    #Tensor(shape=[4, 1], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
    #       [[38.96608230],
    #        [81.28152394],
    #        [69.67229865],
    #        [31.74197251]])
    #Tensor(shape=[4, 4], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
    #       [[0.00000000, 0.00000000, 0.00000000, 0.00000000],
    #        [0.00000000, 0.00000000, 0.00000000, 0.00000000],
    #        [0.00000000, 0.00000000, 0.99998205, 0.00000000],
    #        [0.00000000, 0.00000000, 0.00000000, 0.00000000]])
    ## for rank1 output
    #Tensor(shape=[4, 1], dtype=float64, place=CUDAPlace(1), stop_gradient=True,
    #       [[38.96608230],
    #        [81.28152394],
    #        [69.67229865],
    #        [31.74197251]])
    #Tensor(shape=[4, 8], dtype=float64, place=CUDAPlace(1), stop_gradient=True,
    #       [[0.33943993, 0.00000000, 0.66051859, 0.00000000, 0.00000000, 0.00004148, 0.00000000, 0.00000000],
    #        [0.00000000, 0.00000000, 0.00000000, 0.00000207, 0.99432097, 0.00000000, 0.00567696, 0.00000000],
    #        [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00001795],
    #        [0.00000069, 0.33993085, 0.66006319, 0.00000000, 0.00000000, 0.00000528, 0.00000000, 0.00000000]])