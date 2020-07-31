#
# Some common descriptions used in Paddle API docs
# You can copy the wordings here if that is suitable to your scenario.
#

common_args_en = """
    x (Tensor): : the input tensor, it's data type should be float32, float64, int32, int64.
    name(str, optional): name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
    dtype(str, optional): the data type of the output tensor, can be float32, float64, int32, int64.
    param_attr(ParamAttr, optional): the parameter attribute for learnable weights(Parameter) of this layer.
"""

common_args_cn = """
    x (Tensor) - 输入的Tensor，数据类型为：float32、float64、int32、int64。
    name(str, 可选） - 操作的名称(可选，默认值为None）。 更多信息请参见 :ref:`api_guide_Name`。
    dtype(str, 可选) - 输出Tensor的数据类型，支持int32、int64、float32、float64。
    param_attr(ParamAttr, 可选) – 该Layer的可学习的权重(Parameter)的参数属性。更多信息请参见 :ref:`cn_api_fluid_ParamAttr`。
"""
