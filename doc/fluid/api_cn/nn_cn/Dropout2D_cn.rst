.. _cn_api_nn_Dropout2D:

Dropout2D
-------------------------------

.. py:function:: paddle.nn.Dropout2D(p=0.5, data_format='NCHW', name=None)

根据丢弃概率p，在训练过程中随机将某些通道特征图置0(对一个形状为NCHW的4维张量，通道特征图指的是其中的形状为HW的2维特征图)。
Dropout2D可以提高通道特征图之间的独立性。
论文请参考: [Efficient Object Localization Using Convolutional Networks](https://arxiv.org/abs/1411.4280)。

在动态图模式下，请使用模型的`eval()` 方法切换至测试阶段。

.. note::
   具体实现可参考其对应的功能op: :ref:`cn_api_nn_functional_dropout2d` 。

参数
:::::::::
 - **p** (float): 将输入通道置0的概率， 即丢弃概率。默认: 0.5。
 - **name** (str，可选): 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。
 - **data_format** (str): 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是 `NCHW` 和 `NHWC` 。
                          N是批尺寸，C是通道数，H是特征高度，W是特征宽度。默认值: `NCHW` 。

形状
:::::::::
 - **输入** : 4D `Tensor` 。
 - **输出** : 4D `Tensor` ，形状与输入相同。

代码示例
:::::::::

.. code-block:: python

    import paddle
    import numpy as np

    paddle.disable_static()
    x = np.random.random(size=(2, 3, 4, 5)).astype('float32')
    x = paddle.to_tensor(x)
    m = paddle.nn.Dropout2d(p=0.5)
    y_train = m(x)
    m.eval()  # switch the model to test phase
    y_test = m(x)
    print(x.numpy())
    print(y_train.numpy())
    print(y_test.numpy())
