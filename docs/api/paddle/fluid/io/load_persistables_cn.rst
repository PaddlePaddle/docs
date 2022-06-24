.. _cn_api_fluid_io_load_persistables:

load_persistables
-------------------------------


.. py:function:: paddle.fluid.io.load_persistables(executor, dirname, main_program=None, filename=None)




该接口从给定的 ``main_program`` 中取出所有 ``persistable==True`` 的变量（即持久性变量，详见 :ref:`api_guide_model_save_reader` ），并根据目录 ``dirname``  或 ``filename`` 提供的参数文件对这些持久性变量进行赋值。

使用 ``dirname`` 指定持久性变量的存储路径。若持久性变量以分离文件的形式保存在 ``dirname`` 指定的目录下，则设置 ``filename`` 值为None；若所有持久性变量保存在一个单独的二进制文件中，则使用 ``filename`` 来指明这个二进制文件。

参数
::::::::::::

    - **executor**  (Executor) – 加载持久性变量的 ``executor`` （详见 :ref:`api_guide_executor` ） 。
    - **dirname**  (str) – 持久性变量的存储路径。
    - **main_program**  (Program，可选) – 筛选模型中持久性变量所依据的 ``Program`` （详见 :ref:`api_guide_Program` ）。若为None，则使用全局默认的  ``default_main_program``。默认值为None。
    - **filename**  (str，可选) – 若模型中的持久性变量是以若干文件形式存储在 ``dirname`` 指定的目录下，则设置 ``filename`` 值为None。反之，需要通过 ``filename`` 来指明单一模型持久性变量存储文件的名称。默认值为None。

**返回：** 无
  
代码示例
::::::::::::

COPY-FROM: paddle.fluid.io.load_persistables