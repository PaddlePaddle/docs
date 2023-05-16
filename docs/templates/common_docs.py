#
# Some common descriptions used in Paddle API docs
# You can copy the wordings here if that is suitable to your scenario.
#

common_args_en = """
    x (Tensor): The input tensor, it's data type should be float32, float64, int32, int64.
    y (Tensor): The input tensor, it's data type should be float32, float64, int32, int64.
    name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.
    dtype (str, optional): The data type of the output tensor, can be float32, float64, int32, int64.
    param_attr (ParamAttr, optional): The parameter attribute for learnable weights(Parameter) of this layer. For more information, please refer to :ref:`api_fluid_ParamAttr`.
    bias_attr (ParamAttr, optional): The parameter attribute for learnable bias(Bias) of this layer. For more information, please refer to :ref:`api_fluid_ParamAttr`.
    label (Tensor): The label value corresponding to input, it's data type should be int32, int64.
    learning_rate (Tensor|float): The learning rate, can be a Tensor or a float value. Default is 1e-03.
    axis (int, optional): The axis along which to operate. Default is 0. 
    epsilon (float, optional): Small float added to denominator to avoid dividing by zero. Default is 1e-05.
    is_test (bool, optional): A flag indicating whether execution is in test phase. Default is False, means not in test phase.
    shape (Tensor|tuple|list): Shape of the Tensor. If shape is a list or tuple, the elements of it should be integers or Tensors with shape [1]. If shape is Tensor, it should be an 1-D Tensor .
    keep_dim (bool): Whether to reserve the reduced dimension in the output Tensor. The result tensor will have one fewer dimension than the input unless keep_dim is true. Default is False.
    filter_size (tuple|list|int): The size of convolving kernel. It can be a single integer or a tuple/list containing two integers, representing the height and width of the convolution window respectively. If it is a single integer, the height and width are equal to the integer.
    padding (tuple|int): The padding size. It can be a single integer or a tuple containing two integers, representing the size of padding added to the height and width of the input. If it is a single integer, the both sides of padding are equal to the integer. Default is 0.
    include_sublayers (bool, optional): Whether include the sublayers. If True, return list includes the sublayers weights. Default is True.
    stride (tuple|int): The stride size. It can be a single integer or a tuple containing two integers, representing the strides of the convolution along the height and width. If it is a single integer, the height and width are equal to the integer. Default is 1. 
    groups (int, optional): The group number of convolution layer. When group=n, the input and convolution kernels are divided into n groups equally, the first group of convolution kernels and the first group of inputs are subjected to convolution calculation, the second group of convolution kernels and the second group of inputs are subjected to convolution calculation, ……, the nth group of convolution kernels and the nth group of inputs perform convolution calculations. Default is 1.
    regularization (WeightDecayRegularizer, optional): The strategy of regularization. There are two method: :ref:`api_fluid_regularizer_L1Decay` 、 :ref:`api_fluid_regularizer_L2Decay` . If a parameter has set regularizer using  :ref:`api_fluid_ParamAttr` already, the regularization setting here in optimizer will be ignored for this parameter. Otherwise, the regularization setting here in optimizer will take effect. Default None, meaning there is no regularization.
    grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of some derived class of ``GradientClipBase`` . There are three cliping strategies ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` , :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
    dilation (tuple|int): The dilation size. It can be a single integer or a tuple containing two integers, representing the height and width of dilation of the convolution kernel elements. If it is a single integer,the height and width of dilation are equal to the integer. Default is 1.
    stop_gradient (bool, optional): A boolean that mentions whether gradient should flow. Default is True, means stop calculate gradients.
    force_cpu (bool, optional): Whether force to store the output tensor in CPU memory. If force_cpu is False, the output tensor will be stored in running device memory, otherwise it will be stored  to the CPU memory. Default is False.
    data_format (str, optional): Specify the input data format, the output data format will be consistent with the input, which can be ``NCHW`` or ``NHWC`` . N is batch size, C is channels, H is height, and W is width. Default is ``NCHW`` .
    grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of some derived class of ``GradientClipBase`` . There are three cliping strategies ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` , :ref:`api_fluid_clip_GradientClipByValue` ). Default is None, meaning there is no gradient clipping.
    num_filters (int): The number of filter. It is as same as the output channals numbers.
    dim (int, optional): A dimension along which to operate. Default is 0.
    is_sparse (bool, optional): Whether use sparse updating. For more information, please refer to :ref:`api_guide_sparse_update_en` . If it’s True, it will ues sparse updating.
    place (fluid.CPUPlace()|fluid.CUDAPlace(N)|None): This parameter represents which device the executor runs on, and N means the GPU's id. When this parameter is None, PaddlePaddle will set the default device according to its installation version. If Paddle is CPU version, the default device would be set to CPUPlace(). If Paddle is GPU version, the default device would be set to CUDAPlace(0). Default is None.
    num_filters (int): the number of convolution kernels, is also the number of output channels. 
"""

common_args_cn = """
    x (Tensor) - 输入的 Tensor，数据类型为：float32、float64、int32、int64。
    y (Tensor) - 输入的 Tensor，数据类型为：float32、float64、int32、int64。
    name (str，可选) - 具体用法请参见  :ref:`api_guide_Name` ，一般无需设置，默认值为 None。
    dtype (str，可选) - 输出 Tensor 的数据类型，支持 int32、int64、float32、float64。
    param_attr (ParamAttr，可选) – 该 Layer 的可学习的权重(Parameter)的参数属性。更多信息请参见 :ref:`cn_api_fluid_ParamAttr`。
    bias_attr (ParamAttr，可选) - 该 Layer 的可学习的偏置(Bias)的参数属性。更多信息请参见 :ref:`cn_api_fluid_ParamAttr`。
    label (Tensor) - 训练数据的标签，数据类型为：int32、int64。
    learning_rate (Tensor|float) - 学习率，可以是一个 `Tensor` 或者是一个浮点数。默认值为1e-03.
    axis (int，可选) - 指定对输入 Tensor 进行运算的轴。默认值为0。
    epsilon (float，可选) - 添加到分母上的值以防止分母除0。默认值为1e-05。
    is_test (bool，可选) - 用于表明是否在测试阶段执行。默认值为 False，表示非测试阶段。
    shape (Tensor|tuple|list) - Tensor 的形状。如果 shape 是一个列表或元组，则其元素应该是整数或形状为[]的 0-D Tensor 。 如果 shape 是 Tensor ，则它应该是1-D Tensor。
    keep_dim (bool，可选) - 是否在输出 Tensor 中保留输入的维度。除非 keepdim 为 True，否则输出 Tensor 的维度将比输入 Tensor 小一维，默认值为 False。
    filter_size (tuple|list|int) - 卷积核大小。可以为单个整数或包含两个整数的元组或列表，分别表示卷积核的高和宽。如果为单个整数，表示卷积核的高和宽都等于该整数。
    padding (tuple|int) – 填充大小。可以为单个整数或包含两个整数的元组，分别表示对输入高和宽两侧填充的大小。如果为单个整数，表示高和宽的填充都等于该整数。默认值为0。
    include_sublayers (bool，可选) - 是否返回子层的参数。如果为 True，返回的列表中包含子层的参数。默认值为 True。
    stride (tuple|int) -  步长大小。可以为单个整数或包含两个整数的元组，分别表示卷积沿着高和宽的步长。如果为单个整数，表示沿着高和宽的步长都等于该整数。默认值为1。
    groups (int，可选) - 卷积的组数。当 group=n，输入和卷积核分别平均分为 n 组，第一组卷积核和第一组输入进行卷积计算，第二组卷积核和第二组输入进行卷积计算，……，第 n 组卷积核和第 n 组输入进行卷积计算。默认值为11。
    regularization (WeightDecayRegularizer，可选) - 正则化方法。支持两种正则化策略: :ref:`cn_api_fluid_regularizer_L1Decay` 、 :ref:`cn_api_fluid_regularizer_L2Decay` 。如果一个参数已经在 :ref:`cn_api_fluid_ParamAttr` 中设置了正则化，这里的正则化设置将被忽略；如果没有在 :ref:`cn_api_fluid_ParamAttr` 中设置正则化，这里的设置才会生效。默认值为None，表示没有正则化。
    grad_clip (GradientClipBase，可选) – 梯度裁剪的策略，支持三种裁剪策略： :ref:`cn_api_fluid_clip_GradientClipByGlobalNorm` 、 :ref:`cn_api_fluid_clip_GradientClipByNorm` 、 :ref:`cn_api_fluid_clip_GradientClipByValue` 。
    dilation (tuple|int，可选) - 空洞大小。可以为单个整数或包含两个整数的元组，分别表示卷积核中的元素沿着高和宽的空洞。如果为单个整数，表示高和宽的空洞都等于该整数。默认值为1。
    stop_gradient (bool，可选) - 提示是否应该停止计算梯度，默认值为 True，表示停止计算梯度。
    force_cpu (bool，可选) - 是否强制将输出 Tensor 写入 CPU 内存。如果为 False，则将输出 Tensor 写入当前所在运算设备的内存，否则写入 CPU 内存中。默认为 False。
    data_format (str，可选) - 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是 ``NCHW`` 和 ``NHWC`` 。N 是批大小，C 是通道数，H 是高度，W 是宽度。默认值为 ``NCHW`` 。
    grad_clip (GradientClipBase，可选) – 梯度裁剪的策略，支持三种裁剪策略： :ref:`cn_api_fluid_clip_GradientClipByGlobalNorm` 、 :ref:`cn_api_fluid_clip_GradientClipByNorm` 、 :ref:`cn_api_fluid_clip_GradientClipByValue` 。默认值为 None，表示不使用梯度裁剪。
    num_filters (int) - 卷积核的个数，与输出的通道数相同。
    dim (int，可选) - 指定对输入 Tensor 进行运算的维度。默认值为0。
    is_sparse (bool，可选) - 是否使用稀疏更新的方式，更多信息请参见 :ref:`api_guide_sparse_update` 。默认值为 True，表示使用稀疏更新的方式。
    place (fluid.CPUPlace()|fluid.CUDAPlace(N)|None) – 该参数表示 Executor 执行所在的设备，这里的 N 为 GPU 对应的 ID。当该参数为 None 时，PaddlePaddle 会根据其安装版本来设置默认设备。当 PaddlePaddle 是 CPU 版时，默认运行设备将会设置为 ``fluid.CPUPlace()`` ；当 PaddlePaddle 是 GPU 版本时，默认执行设备将会设置为 ``fluid.CUDAPlace(0)`` 。默认值为 None。
    num_filters (int) - 卷积核个数，同时也是输出的通道数。
"""
