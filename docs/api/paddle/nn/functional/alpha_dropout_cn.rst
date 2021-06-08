.. _cn_api_nn_functional_alpha_dropout:

alpha_dropout
-------------------------------

.. py:function:: paddle.nn.functional.alpha_dropout(x, p=0.5, training=True, name=None)

alpha_dropout是一种具有自归一化性质的dropout。均值为0，方差为1的输入，经过alpha_dropout计算之后，输出的均值和方差与输入保持一致。alpha_dropout通常与SELU激活函数组合使用。

参数
:::::::::
 - **x** (Tensor): 输入的多维 `Tensor` ，数据类型为：float32、float64。
 - **p** (float): 将输入节点置0的概率，即丢弃概率。默认: 0.5。
 - **training** (bool): 标记是否为训练阶段。 默认: True。
 - **name** (str，可选): 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

返回
:::::::::
经过alpha_dropout之后的结果，与输入x形状相同的 `Tensor` 。

代码示例
:::::::::

.. code-block:: python

    import paddle
    import numpy as np

    x = np.array([[-1, 1], [-1, 1]]).astype('float32')
    x = paddle.to_tensor(x)
    y_train = paddle.nn.functional.alpha_dropout(x, 0.5)
    y_test = paddle.nn.functional.alpha_dropout(x, 0.5, training=False)
    print(x)
    print(y_train)
    # [[-0.10721093, 1.6655989 ], [-0.7791938, -0.7791938]] (randomly)
    print(y_test)
