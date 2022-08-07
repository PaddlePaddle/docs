.. _cn_api_paddle_nn_Conv1D:

Conv1D
-------------------------------

.. py:class:: paddle.nn.Conv1D(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', weight_attr=None, bias_attr=None, data_format="NCL")


返回一个用于计算一维卷积的类实例，它是 :ref:`cn_api_nn_functional_conv1d` 的一个特殊情形。


参数
::::::::::::
    - **in_channels** (int) - 输入特征的通道数。
    - **out_channels** (int) - 输入特征的通道数。
    - **kernel_size** (int|list|tuple) - 卷积核大小，与 :ref:`cn_api_nn_functional_conv1d` 的同名参数一致。
    - **stride** (int|list|tuple，可选) - 步长大小，与 :ref:`cn_api_nn_functional_conv1d` 的同名参数一致。默认值：1。
    - **padding** (int|list|tuple|str，可选) - 填充大小，与 :ref:`cn_api_nn_functional_conv1d` 的同名参数一致。默认值：0。
    - **dilation** (int|list|tuple，可选) - 空洞大小，与 :ref:`cn_api_nn_functional_conv1d` 的同名参数一致。默认值：1。
    - **groups** (int，可选) - 分组卷积的组数，与 :ref:`cn_api_nn_functional_conv1d` 的同名参数一致。默认值：1。
    - **padding_mode** (str，可选)：填充模式，与 :ref:`cn_api_nn_functional_conv1d` 的同名参数一致。默认值：``'zeros'`` 。
    - **weight_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值为 None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **bias_attr** （ParamAttr|bool，可选）- 指定偏置参数属性的对象。若 ``bias_attr`` 为 bool 类型，只支持为 False，表示没有偏置参数。默认值为 None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **data_format** (str，可选) - 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCL"和"NLC"。N 是批尺寸，C 是通道数，L 是特征长度。默认值："NCL"。

    
可学习参数
::::::::::::
    - **Conv1D.weight** (Tensor) - 与 :ref:`cn_api_nn_functional_conv1d` 的同名参数对应，形状为 :math:`[\mathrm{out\_channels}, \mathrm{in\_channels}/\mathrm{groups},\mathrm{kernel\_size}]`。
    - **Conv1D.bias** (Tensor) - 与 :ref:`cn_api_nn_functional_conv1d` 的同名参数对应，形状为 :math:`[\mathrm{out\_channels}]`，在 :attr:`bias_attr` 为 False 时所有元素的值都是 :math:`0.0`。


代码示例
::::::::::::

COPY-FROM: paddle.nn.Conv1D
