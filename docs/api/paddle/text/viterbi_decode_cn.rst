.. _cn_api_paddle_text_viterbi_decode:

viterbi_decode
-------------------------------
.. py:function:: paddle.text.viterbi_decode(potentials, transition_params, lengths, include_bos_eos_tag=True, name=None)

该层利用输入的发射概率和转移概率进行解码。通过用Viterbi算法，动态地寻找隐藏状态最可能的序列，该序列也被称为 Viterbi 路径（Viterbi path），从而得到观察标签 (tags) 序列。

参数
:::::::::
    - **potentials (Tensor)** 发射概率。形状为[batch_size, lengths, num_tags]，数据类型为float32或float64。
    - **transition_params (Tensor)** 转移概率。形状为[num_tags, num_tags]，数据类型为float32或float64。
    - **lengths (Tensor)** 序列真实长度。形状为[batch_size]，数据类型为int64。
    - **include_bos_eos_tag (bool, 可选)** 是否包含前置、后置标签。如果设为True，**transition_params** 中倒数第一列为前置标签的转移概率，倒数第二列为后置标签的转移概率。默认值为True。
    - **name (str, 可选）** 默认值为None。一般用户无需设置，具体用法请参见 :ref:`api_guide_Name` 。

返回
:::::::::
    - **scores (Tensor)** Viterbi路径的最高得分。形状为[batch_size]，数据类型为float32或float64。
    - **paths (Tensor)** Viterbi路径。形状为[batch_size, lengths]，数据类型为int64。

代码示例
:::::::::

..  code-block:: python

    import numpy as np
    import paddle
    paddle.seed(102)
    batch_size, seq_len, num_tags = 2, 4, 3
    emission = paddle.rand((batch_size, seq_len, num_tags), dtype='float32')
    length = paddle.randint(1, seq_len + 1, [batch_size])
    tags = paddle.randint(0, num_tags, [batch_size, seq_len])
    transition = paddle.rand((num_tags, num_tags), dtype='float32')
    scores, path = paddle.text.viterbi_decode(emission, transition, length, False)
    # scores: Tensor(shape=[2], dtype=float32, place=CUDAPlace(0), stop_gradient=True, [3.37089300, 1.56825531])
    # path: Tensor(shape=[2, 3], dtype=int64, place=CUDAPlace(0), stop_gradient=True, [[1, 0, 0], [1, 1, 0]])