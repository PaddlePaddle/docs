#
# Some common descriptions used in Paddle API docs
# You can copy the wordings here if that is suitable to your scenario.
#

common_args_en = """
    x (Tensor): The input tensor, it's data type should be float32, float64, int32, int64.
    y (Tensor): The input tensor, it's data type should be float32, float64, int32, int64.
    name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
    dtype (str, optional): The data type of the output tensor, can be float32, float64, int32, int64.
    param_attr (ParamAttr, optional): The parameter attribute for learnable weights(Parameter) of this layer. For more information, please refer to :ref:`api_fluid_ParamAttr`.
    bias_attr (ParamAttr, optional): The parameter attribute for learnable bias(Bias) of this layer. For more information, please refer to :ref:`api_fluid_ParamAttr`.
    label (Tensor): The label value corresponding to input, it's data type should be int32, int64.
    learning_rate (Tensor|float): The learning rate, can be a Tensor or a float value. Default is 1e-03.
    axis (int, optional): The axis to calculate the input tensor. Default is 0. 
    epsilon (float, optional): Small float added to denominator to avoid dividing by zero. Default is 1e-05.
    is_test (bool, optional): A flag indicating whether execution is in test phase. Default is False, means not in test phase.
    shape (Tensor|tuple|list): Shape of the Tensor. If shape is a list or tuple, the elements of it should be integers or Tensors with shape [1]. If shape is Tensor, it should be an 1-D Tensor .
    keep_dim (bool): Whether to reserve the reduced dimension in the output Tensor. The result tensor will have one fewer dimension than the input unless keep_dim is true. Default is False.
    filter_size (tuple|list|int): The size of convolving kernel. It can be a single integer or a tuple/list containing two integers, representing the height and width of the convolution window respectively. If it is a single integer, the height and width are equal to the integer.
    padding (tuple|int): The padding size. It can be a single integer or a tuple containing two integers, representing the size of padding added to the height and width of the input. If it is a single integer, the both sides of padding are equal to the integer. Default is 0.
    include_sublayers (bool, optional): Whether include the sublayers. If True, return list includes the sublayers weights. Default is True.
    stride (tuple|int): The stride size. It can be a single integer or a tuple containing two integers, representing the strides of the convolution along the height and width. If it is a single integer, the height and width are equal to the integer. Default is 1. 
    groups (int, optional): The group number of convolution layer. When group=n, the input and convolution kernels are divided into n groups equally, the first group of convolution kernels and the first group of inputs are subjected to convolution calculation, the second group of convolution kernels and the second group of inputs are subjected to convolution calculation, ……, the nth group of convolution kernels and the nth group of inputs perform convolution calculations. Default is 1.
    regularization (WeightDecayRegularizer, optional) – The strategy of regularization. There are two method: :ref:`api_fluid_regularizer_L1Decay` 、 :ref:`api_fluid_regularizer_L2Decay` . If a parameter has set regularizer using  :ref:`api_fluid_ParamAttr` already, the regularization setting here in optimizer will be ignored for this parameter. Otherwise, the regularization setting here in optimizer will take effect. Default None, meaning there is no regularization.
    grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of some derived class of ``GradientClipBase`` . There are three cliping strategies ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` , :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
    dilation (tuple|int) – The dilation size. It can be a single integer or a tuple containing two integers, representing the height and width of dilation of the convolution kernel elements. If it is a single integer,the height and width of dilation are equal to the integer. Default is 1.
    stop_gradient (bool, optional) – A boolean that mentions whether gradient should flow. Default is True, means stop calculate gradients.
"""

common_args_cn = """
    x (Tensor) - 输入的Tensor，数据类型为：float32、float64、int32、int64。
    y (Tensor) - 输入的Tensor，数据类型为：float32、float64、int32、int64。
    name (str, 可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。
    dtype (str, 可选) - 输出Tensor的数据类型，支持int32、int64、float32、float64。
    param_attr (ParamAttr, 可选) – 该Layer的可学习的权重(Parameter)的参数属性。更多信息请参见 :ref:`cn_api_fluid_ParamAttr`。
    bias_attr (ParamAttr, 可选) - 该Layer的可学习的偏置(Bias)的参数属性。更多信息请参见 :ref:`cn_api_fluid_ParamAttr`。
    label (Tensor) - 训练数据的标签，数据类型为：int32, int64。
    learning_rate (Tensor|float) - 学习率，可以是一个Tensor或者是一个浮点数。默认值为1e-03.
    axis (int, 可选) - 指定对输入Tensor进行运算的轴。默认值为0。
    epsilon (float, 可选) - 添加到分母上的值以防止分母除0。默认值为1e-05。
    is_test (bool, 可选) - 用于表明是否在测试阶段执行。默认值为False，表示非测试阶段。
    shape (Tensor|tuple|list) - Tensor的形状。如果shape是一个列表或元组，则其元素应该是形状为[1]的整数或Tensor。 如果shape是Tensor，则它应该是一维Tensor。
    keep_dim (bool) - 是否在输出Tensor中保留减小的维度。如 keep_dim 为True，否则结果张量的维度将比输入张量小，默认值为False。
    filter_size (tuple|list|int) - 卷积核大小。可以为单个整数或包含两个整数的元组或列表，分别表示卷积核的高和宽。如果为单个整数，表示卷积核的高和宽都等于该整数。
    padding (tuple|int) – 填充大小。可以为单个整数或包含两个整数的元组，分别表示对输入高和宽两侧填充的大小。如果为单个整数，表示高和宽的填充都等于该整数。默认值为0。
    include_sublayers (bool, 可选) - 是否返回子层的参数。如果为True，返回的列表中包含子层的参数。默认值为True。
    stride (tuple|int) -  步长大小。可以为单个整数或包含两个整数的元组，分别表示卷积沿着高和宽的步长。如果为单个整数，表示沿着高和宽的步长都等于该整数。默认值为1。
    groups (int, 可选) - 卷积的组数。当group=n，输入和卷积核分别平均分为n组，第一组卷积核和第一组输入进行卷积计算，第二组卷积核和第二组输入进行卷积计算，……，第n组卷积核和第n组输入进行卷积计算。默认值为11。
    regularization (WeightDecayRegularizer，可选) - 正则化方法。支持两种正则化策略: :ref:`cn_api_fluid_regularizer_L1Decay` 、 :ref:`cn_api_fluid_regularizer_L2Decay` 。如果一个参数已经在 :ref:`cn_api_fluid_ParamAttr` 中设置了正则化，这里的正则化设置将被忽略；如果没有在 :ref:`cn_api_fluid_ParamAttr` 中设置正则化，这里的设置才会生效。默认值为None，表示没有正则化。
    grad_clip (GradientClipBase, 可选) – 梯度裁剪的策略，支持三种裁剪策略： :ref:`cn_api_fluid_clip_GradientClipByGlobalNorm` 、 :ref:`cn_api_fluid_clip_GradientClipByNorm` 、 :ref:`cn_api_fluid_clip_GradientClipByValue` 。
    dilation (tuple|int, 可选) - 空洞大小。可以为单个整数或包含两个整数的元组，分别表示卷积核中的元素沿着高和宽的空洞。如果为单个整数，表示高和宽的空洞都等于该整数。默认值为1。
    stop_gradient (bool，可选) - 提示是否应该停止计算梯度，默认值为True，表示停止计算梯度。
"""
