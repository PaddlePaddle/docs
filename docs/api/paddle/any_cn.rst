.. _cn_api_tensor_any:

any
-------------------------------

.. py:function:: paddle.any(x, axis=None, keepdim=False, name=None)

该OP是对指定维度上的Tensor元素进行逻辑或运算，并输出相应的计算结果。

参数
:::::::::
    - **x** （Tensor）- 输入变量为多维Tensor，数据类型为bool。
    - **axis** （int | list | tuple ，可选）- 计算逻辑或运算的维度。如果为None，则计算所有元素的逻辑或并返回包含单个元素的Tensor变量，否则必须在  :math:`[−rank(x),rank(x)]` 范围内。如果 :math:`axis [i] <0` ，则维度将变为 :math:`rank+axis[i]` ，默认值为None。
    - **keepdim** （bool）- 是否在输出Tensor中保留减小的维度。如 keepdim 为true，否则结果张量的维度将比输入张量小，默认值为False。
    - **name** （str ， 可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回
:::::::::
  Tensor，在指定维度上进行逻辑或运算的Tensor，数据类型和输入数据类型一致。


代码示例
:::::::::

COPY-FROM: <any>:<code-example1>
