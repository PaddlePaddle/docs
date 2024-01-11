.. _cn_api_paddle_bitwise_left_shift:

bitwise_left_shift
-------------------------------

.. py:function:: paddle.bitwise_left_shift(x, y, is_arithmetic=True, out=None, name=None)

对 Tensor ``x`` 和 ``y`` 逐元素进行 ``按位算术(或逻辑)左移`` 运算。

关于 **有符号数的符号位** 在不同情景下的行为：
  1. 算术左移时，符号位同其他位一样，一起左移，右边补0；
  2. 逻辑左移时，符号位同其他位一样，一起左移，右边补0；
  3. 算术右移时，符号位同其他位一样，一起右移，左边补符号位；
  4. 逻辑右移时，符号位同其他位一样，一起右移，左边补0；

.. note::
    当有符号数左移发生溢出时，其值不可控，可能会在左移时突然变号，这是因为在左移时，有符号数的符号位同样进行左移，会导致符号位右侧的值不断成为符号位，例如

    example1:

    .. code-block:: text

        int8_t x = -45; // 补码为 1101,0011      表示-45

        int8_t y = x << 2;   //补码为 0100,1100  表示76

        int8_t z = x << 3;   //补码为 1001,1000  表示-104

    example2:

    .. code-block:: text

        int8_t x = -86; // 补码为 1010,1010      表示-86

        int8_t y = x << 1;   //补码为 0101,0100  表示84

        int8_t z = x << 2;   //补码为 1010,1000  表示-88

    以上为溢出导致的符号突变。

.. math::
        Out = X \ll Y

.. note::
    ``paddle.bitwise_left_shift`` 遵守 broadcasting，如您想了解更多，请参见 `Tensor 介绍`_ .

    .. _Tensor 介绍: ../../guides/beginner/tensor_cn.html#id7
参数
::::::::::::

        - **x** （Tensor）- 输入的 N-D `Tensor`，数据类型为：uint8，int8，int16，int32，int64。
        - **y** （Tensor）- 输入的 N-D `Tensor`，数据类型为：uint8，int8，int16，int32，int64。
        - **is_arithmetic** （bool） - 用于表明是否执行算术位移，True 表示算术位移，False 表示逻辑位移。默认值为 True，表示算术位移。
        - **out** （Tensor，可选）- 输出的结果 `Tensor`，是与输入数据类型相同的 N-D `Tensor`。默认值为 None，此时将创建新的 Tensor 来保存输出结果。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::
 ``按位算术(逻辑)左移`` 运算后的结果 ``Tensor``，数据类型与 ``x`` 相同。

代码示例1
::::::::::::

算术左移

COPY-FROM: paddle.bitwise_left_shift:bitwise_left_shift_example1

代码示例2
::::::::::::

逻辑左移

COPY-FROM: paddle.bitwise_left_shift:bitwise_left_shift_example2