.. _cn_api_nn_Pad1D:

Pad1D
-------------------------------
.. py:class:: paddle.nn.Pad1D(padding, mode="constant", value=0.0, data_format="NCL", name=None)

此接口用于构造Pad1D类的可调用对象，按照 padding、mode 和 value 属性对输入进行填充。pad[0] 和 pad[1] 不能大于 width-1。

参数
::::::::::::

  - **padding** (Tensor|list[int]|int) - 填充大小。如果是 int，则在所有待填充边界使用相同的填充，否则填充的格式为[pad_left, pad_right]。
  - **mode** (str，可选) - padding 的四种模式，分别为 ``'constant'``、``'reflect'``、``'replicate'`` 和 ``'circular'``，默认值为 ``'constant'``。

     - ``'constant'`` 表示填充常数 ``value``；
     - ``'reflect'`` 表示填充以输入边界值为轴的映射；
     - ``'replicate'`` 表示填充输入边界值；
     - ``'circular'`` 为循环填充输入。

  - **value** (float，可选) - 以 ``'constant'`` 模式填充区域时填充的值。默认值为 :math:`0.0`。
  - **data_format** (str，可选)  - 指定输入的数据格式，可为 ``'NCL'`` 或者 ``'NLC'`` ，默认值为 ``'NCL'``。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
无

代码示例
::::::::::::

COPY-FROM: paddle.nn.Pad1D

输出 (input, label)
:::::::::
    定义每次调用时执行的计算。应被所有子类覆盖。

参数
:::::::::
    - **inputs** (tuple) - 未压缩的 tuple 参数。
    - **kwargs** (dict) - 未压缩的字典参数。

extra_repr()
:::::::::
    该层为额外层，您可以自定义实现层。