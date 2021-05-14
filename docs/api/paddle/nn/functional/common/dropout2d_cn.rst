.. _cn_api_nn_functional_dropout2d:

dropout2d
-------------------------------

.. py:function:: paddle.nn.functional.dropout2d(x, p=0.5, training=True, name=None)

该算子根据丢弃概率 `p` ，在训练过程中随机将某些通道特征图置0(对一个形状为 `NCHW` 的4维张量，通道特征图指的是其中的形状为 `HW` 的2维特征图)。

.. note::
   该op基于 ``paddle.nn.functional.dropout`` 实现，如您想了解更多，请参见 :ref:`cn_api_nn_functional_dropout` 。

参数
:::::::::
 - **x** (Tensor): 形状为[N, C, H, W]或[N, H, W, C]的4D `Tensor` ，数据类型为float32或float64。
 - **p** (float): 将输入通道置0的概率，即丢弃概率。默认: 0.5。
 - **training** (bool): 标记是否为训练阶段。 默认: True。
 - **name** (str，可选): 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

返回
:::::::::
经过dropout2d之后的结果，与输入x形状相同的 `Tensor` 。

代码示例
:::::::::

.. code-block:: python

    import paddle
    import numpy as np

    x = np.random.random(size=(2, 3, 4, 5)).astype('float32')
    x = paddle.to_tensor(x)
    y_train = paddle.nn.functional.dropout2d(x)  #train
    y_test = paddle.nn.functional.dropout2d(x, training=False) 
    for i in range(2):
        for j in range(3):
            print(x.numpy()[i,j,:,:])
            print(y_train.numpy()[i,j,:,:])
            print(y_test.numpy()[i,j,:,:])
