.. _cn_api_incubate_nn_functional_fused_feedforward:

fused_feedforward
-------------------------------

.. py:function:: paddle.incubate.nn.functional.fused_feedforward(x, linear1_weight, linear2_weight, linear1_bias=None, linear2_bias=None, ln1_scale=None, ln1_bias=None, ln2_scale=None, ln2_bias=None, dropout1_rate=0.5, dropout2_rate=0.5,activation="relu", ln1_epsilon=1e-5, ln2_epsilon=1e-5, pre_layer_norm=False, training=True, mode='upscale_in_train', name=None):

这是一个融合算子，该算子是对transformer模型中feed forward层的多个算子进行融合，该算子只支持在GPU下运行，该算子与如下伪代码表达一样的功能：

.. code-block:: ipython

    residual = src;
    if pre_layer_norm:
        src = layer_norm(src)
    src = linear(dropout(activation(linear(src))))
    src = residual + dropout(src)
    if not pre_layer_norm:
        src = layer_norm(src)

参数
:::::::::
    - **x** (Tensor) - 输入Tensor，数据类型支持float16， float32 和float64，输入的形状是 ``[batch_size, sequence_length, d_model]`` 。
    - **linear1_weight** (Tensor) - 第一个linear算子的权重数据，数据类型与 ``x`` 一样，形状是 ``[d_model, dim_feedforward]`` 。
    - **linear2_weight** (Tensor) - 第二个linear算子的权重数据，数据类型与 ``x`` 一样，形状是 ``[dim_feedforward, d_model]`` 。
    - **linear1_bias** (Tensor，可选) - 第一个linear算子的偏置数据，数据类型与 ``x`` 一样，形状是 ``[dim_feedforward]``。默认值为None。
    - **linear2_bias** (Tensor，可选) - 第二个linear算子的偏置数据，数据类型与 ``x`` 一样，形状是 ``[d_model]``。默认值为None。
    - **ln1_scale** (Tensor，可选) - 第一个layer_norm算子的权重数据，数据类型可以是float32或者float64，形状和 ``x`` 一样。默认值为None。
    - **ln1_bias** (Tensor，可选) - 第一个layer_norm算子的偏置数据，数据类型和 ``ln1_scale`` 一样，形状是 ``x.shape[-1]``。默认值为None。
    - **ln2_scale** (Tensor，可选) - 第二个layer_norm算子的权重数据，数据类型可以是float32或者float64，形状和  ``x`` 一样。默认值为None。
    - **ln2_bias** (Tensor，可选) - 第二个layer_norm算子的偏置数据，数据类型和 ``ln2_scale`` 一样，形状是 ``x.shape[-1]``。默认值为None。
    - **dropout1_rate** (float，可选) - 第一个dropout算子置零的概率。默认是0.5。
    - **dropout2_rate** (float，可选) - 第二个dropout算子置零的概率。默认是0.5。
    - **activation** (string，可选) - 激活函数，当前只支持relu和gelu。默认值是relu。
    - **ln1_epsilon** (float，可选) - 一个很小的浮点数，被第一个layer_norm算子加到分母，避免出现除零的情况。默认值是1e-5。
    - **ln2_epsilon** (float，可选) - 一个很小的浮点数，被第二个layer_norm算子加到分母，避免出现除零的情况。默认值是1e-5。
    - **pre_layer_norm** (bool，可选) - 在预处理阶段加上layer_norm，或者在后处理阶段加上layer_norm。默认值是False。
    - **training** (bool)：标记是否为训练阶段。默认：True。
    - **mode** (str)：丢弃单元的方式，有两种'upscale_in_train'和'downscale_in_infer'，默认：'upscale_in_train'。计算方法如下：

        1. upscale_in_train，在训练时增大输出结果。

            - train: out = input * mask / ( 1.0 - p )
            - inference: out = input

        2. downscale_in_infer，在预测时减小输出结果

            - train: out = input * mask
            - inference: out = input * (1.0 - p)

    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
    - Tensor，输出Tensor，数据类型与 ``x`` 一样。

代码示例
::::::::::

.. code-block:: python

    # required: gpu
    import paddle
    import numpy as np
    x_data = np.random.random((1, 8, 8)).astype("float32")
    linear1_weight_data = np.random.random((8, 8)).astype("float32")
    linear2_weight_data = np.random.random((8, 8)).astype("float32")
    x = paddle.to_tensor(x_data)
    linear1_weight = paddle.to_tensor(linear1_weight_data)
    linear2_weight = paddle.to_tensor(linear2_weight_data)
    out = paddle.incubate.nn.functional.fused_feedforward(x, linear1_weight, linear2_weight)
    print(out.numpy().shape)
    # (1, 8, 8)

