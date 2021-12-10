.. _cn_api_paddle_tensor_lcm:

lcm
-------------------------------

.. py:function:: paddle.lcm(x, y, name=None)

计算两个输入的按元素绝对值的最小公倍数，输入必须是整型。

.. note::

    lcm(0,0)=0, lcm(0, y)=0

    如果x和y的shape不一致，会对两个shape进行广播操作，得到一致的shape（并作为输出结果的shape），
    请参见 :ref:`cn_user_guide_broadcasting` 。

参数
:::::::::

- **x**  (Tensor) - 输入的Tensor，数据类型为：int8，int16，int32，int64，uint8。
- **y**  (Tensor) - 输入的Tensor，数据类型为：int8，int16，int32，int64，uint8。
- **name**  (str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
:::::::::

输出Tensor，与输入数据类型相同。

代码示例
:::::::::

COPY-FROM: paddle.lcm
