.. _cn_api_nn_Dropout:

Dropout
-------------------------------

.. py:function:: paddle.nn.Dropout(p=0.5, axis=None, mode="upscale_in_train”, name=None)

Dropout是一种正则化手段，该算子根据给定的丢弃概率 `p` ，在训练过程中随机将一些神经元输出设置为0，通过阻止神经元节点间的相关性来减少过拟合。论文请参考: `Improving neural networks by preventing co-adaptation of feature detectors <https://arxiv.org/abs/1207.0580>`_ 

在动态图模式下，请使用模型的 `eval()` 方法切换至测试阶段。

.. note::
   对应的 `functional方法` 请参考: :ref:`cn_api_nn_functional_dropout` 。

参数
:::::::::
 - **p** (float): 将输入节点置为0的概率， 即丢弃概率。默认: 0.5。
 - **axis** (int|list): 指定对输入 `Tensor` 进行Dropout操作的轴。默认: None。
 - **mode** (str): 丢弃单元的方式，有两种'upscale_in_train'和'downscale_in_infer'，默认: 'upscale_in_train'。计算方法如下:

    1. upscale_in_train, 在训练时增大输出结果。

       - train: out = input * mask / ( 1.0 - p )
       - inference: out = input

    2. downscale_in_infer, 在预测时减小输出结果

       - train: out = input * mask
       - inference: out = input * (1.0 - p)

 - **name** (str，可选): 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

形状
:::::::::
 - **输入** : N-D `Tensor` 。
 - **输出** : N-D `Tensor` ，形状与输入相同。

代码示例
:::::::::

.. code-block:: python

    import paddle
    import numpy as np

    paddle.disable_static()
    x = np.array([[1,2,3], [4,5,6]]).astype('float32')
    x = paddle.to_tensor(x)
    m = paddle.nn.Dropout(p=0.5)
    y_train = m(x)
    m.eval()  # switch the model to test phase
    y_test = m(x)
    print(x.numpy())
    print(y_train.numpy())
    print(y_test.numpy())
