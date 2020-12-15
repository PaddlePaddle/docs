.. _cn_api_nn_Dropout3D:

Dropout3D
-------------------------------

.. py:function:: paddle.nn.Dropout3D(p=0.5, data_format='NCDHW', name=None)

根据丢弃概率 `p` ，在训练过程中随机将某些通道特征图置0(对一个形状为 `NCDHW` 的5维张量，通道特征图指的是其中的形状为 `DHW` 的3维特征图)。Dropout3D可以提高通道特征图之间的独立性。论文请参考: `Efficient Object Localization Using Convolutional Networks <https://arxiv.org/abs/1411.4280>`_

在动态图模式下，请使用模型的 `eval()` 方法切换至测试阶段。

.. note::
   对应的 `functional方法` 请参考: :ref:`cn_api_nn_functional_dropout3d` 。

参数
:::::::::
 - **p** (float): 将输入通道置0的概率， 即丢弃概率。默认: 0.5。
 - **data_format** (str): 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是 `NCDHW` 和 `NDHWC` 。其中 `N` 是批尺寸， `C` 是通道数， `D` 是特征深度， `H` 是特征高度， `W` 是特征宽度。默认值: `NCDHW` 。
 - **name** (str，可选): 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

形状
:::::::::
 - **输入** : 5-D `Tensor` 。
 - **输出** : 5-D `Tensor` ，形状与输入相同。

代码示例
:::::::::

.. code-block:: python

    import paddle
    import numpy as np

    paddle.disable_static()
    x = np.random.random(size=(2, 3, 4, 5, 6)).astype('float32')
    x = paddle.to_tensor(x)
    m = paddle.nn.Dropout3D(p=0.5)
    y_train = m(x)
    m.eval()  # switch the model to test phase
    y_test = m(x)
    print(x.numpy())
    print(y_train.numpy())
    print(y_test.numpy())
