.. _cn_api_fluid_initializer_ConstantInitializer:

ConstantInitializer
-------------------------------

.. py:class:: paddle.fluid.initializer.ConstantInitializer(value=0.0, force_cpu=False)




该接口为常量初始化函数，用于权重初始化，通过输入的value值初始化输入变量；

参数
::::::::::::

        - **value** (float16|float32) - 用于初始化输入变量的值；
        - **force_cpu** (bool) - 用于表示初始化的位置，为True时，会强制在CPU上初始化（即使executor设置的是CUDA）。默认为False。

返回
::::::::::::
参数初始化类的实例

代码示例
::::::::::::

COPY-FROM: paddle.fluid.initializer.ConstantInitializer