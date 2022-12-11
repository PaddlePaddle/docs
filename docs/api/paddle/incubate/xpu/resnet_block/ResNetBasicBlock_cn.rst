.. _cn_api_incubate_xpu_ResNetBasicBlock:

ResNetBasicBlock
-------------------------------
.. py:class:: paddle.incubate.xpu.ResNetBasicBlock(num_channels1, num_filter1, filter1_size, num_channels2, num_filter2, filter2_size, num_channels3, num_filter3, filter3_size, stride1=1, stride2=1, stride3=1, act='relu', momentum=0.9, eps=1e-5, data_format='NCHW', has_shortcut=False, use_global_stats=False, is_test=False, filter1_attr=None, scale1_attr=None, bias1_attr=None, moving_mean1_name=None, moving_var1_name=None, filter2_attr=None, scale2_attr=None, bias2_attr=None, moving_mean2_name=None, moving_var2_name=None, ilter3_attr=None, scale3_attr=None, bias3_attr=None, moving_mean3_name=None, moving_var3_name=None, padding1=0, padding2=0, padding3=0, dilation1=1, dilation2=1, dilation3=1, trainable_statistics=False, find_conv_max=True)

该接口用于构建 ``ResNetBasicBlock`` 类的一个可调用对象，实现一次性计算多个 ``Conv2D``、 ``BatchNorm`` 和 ``ReLU`` 的功能，排列顺序参见源码链接。

当 has_shortcut = False 时，实现计算两个 ``Conv2D``、两个 ``BatchNorm`` 和两个 ``ReLU`` 的功能，此时输入输出 ``Tensor`` 的 shape 需要保持一致。

当 has_shortcut = True 时，实现计算三个 ``Conv2D``、三个 ``BatchNorm`` 和两个 ``ReLU`` 的功能。



参数
:::::::::
    - **num_channels** (int) - 输入图像的通道数。
    - **num_filter** (int) - 由卷积操作产生的输出的通道数。
    - **filter_size** (int|list|tuple) - 卷积核大小。可以为单个整数或包含两个整数的元组或列表，分别表示卷积核的高和宽。如果为单个整数，表示卷积核的高和宽都等于该整数。
    - **stride** (int，可选) - 步长大小。为单个整数，表示沿着高和宽的步长都等于该整数。默认值：1。
    - **act** (str，可选) - 激活函数。默认值： ``relu``。
    - **momentum** (float，可选) - 此值用于计算 ``moving_mean`` 和 ``moving_var``。默认值：0.9。
    - **eps** (float，可选) - 为了数值稳定加在分母上的值。默认值：1e-5。
    - **data_format** (str，可选) - 指定输入的数据格式，输出的数据格式将与输入保持一致，目前只支持"NCHW"。N 是批尺寸，C 是通道数，H 是特征高度，W 是特征宽度。默认值："NCHW"。
    - **has_shortcut** (bool，可选) - 设置是否计算第三个 Conv2D 和 BatchNorm。为 True 时计算，否则不进行计算。默认值： ``False`` 。
    - **use_global_stats** (bool，可选) – 指示是否使用全局均值和方差。在预测或测试模式下，将 ``use_global_stats`` 设置为 true 或将 ``is_test`` 设置为 true，这两种行为是等效的。在训练模式中，当设置 ``use_global_stats`` 为 True 时，在训练期间也将使用全局均值和方差。默认值：False。
    - **is_test** (bool，可选) - 指示是否在测试阶段，非训练阶段使用训练过程中统计到的全局均值和全局方差。默认值：False。
    - **filter_attr** (ParamAttr，可选) - 指定对应 ``Conv2D`` 权重参数属性的对象。默认值为 None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **scale_attr** (ParamAttr，可选) - 指定对应 ``BatchNorm`` 权重参数属性的对象。默认值为 None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **bias_attr** (ParamAttr，可选) - 指定对应 ``BatchNorm`` 偏置参数属性的对象。默认值为 None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **moving_mean_name** (str，可选) - ``moving_mean`` 的名称，存储全局均值。默认值：None。
    - **moving_var_name** (str，可选) - ``moving_var`` 的名称，存储全局方差。默认值：None。
    - **padding** (int，可选) - 填充大小。为一个整数，padding_height = padding_width = padding。默认值：0。
    - **dilation** (int，可选) - 空洞大小。为单个整数，表示高和宽的空洞都等于该整数。默认值：1。
    - **trainable_statistics** (bool，可选) - eval 模式下是否计算 mean 均值和 var 方差。eval 模式下，trainable_statistics 为 True 时，由该批数据计算均值和方差。默认值：False。
    - **find_conv_max** (bool，可选) - 是否计算每个 Conv2D 输入 ``Tensor`` 的最大值，为 True 表示计算。默认值：True。


返回
:::::::::
    - Tensor，输出 Tensor，数据类型与 ``X`` 一样。



代码示例
::::::::::

COPY-FROM: paddle.incubate.xpu.resnet_block.ResNetBasicBlock
