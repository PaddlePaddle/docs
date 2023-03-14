.. _cn_api_paddle_polar:

polar
-------------------------------

.. py:function:: paddle.polar(abs, angle, name=None)


对于给定的模 ``abs`` 和相位角 ``angle``，返回一个对应复平面（笛卡尔坐标系）上的复数坐标 Tensor。

.. math::
    \text{out} = \text{abs}\cdot\cos(\text{angle}) + \text{abs}\cdot\sin(\text{angle})\cdot j

参数
::::::::::
    - **abs** (Tensor) – 输入是一个多维的 Tensor，它的数据类型可以是 float32，float64。
    - **angle** (Tensor) – 输入是一个多维的 Tensor，它的数据类型可以是 float32，float64。
    - **name** (str，可选) - 具体用法请参见  :ref:`api_guide_Name` ，一般无需设置，默认值为 None。
返回
::::::::::
    - ``Tensor`` (Tensor)：对应模 ``abs`` 和相位角 ``angle`` 在复平面（笛卡尔坐标系）上的复数坐标 Tensor，形状和原输入的形状一致。


代码示例
::::::::::

COPY-FROM: paddle.polar
