.. _cn_api_fluid_io_load_params:

load_params
-------------------------------


.. py:function:: paddle.fluid.io.load_params(executor, dirname, main_program=None, filename=None)




该接口从指定的 ``main_program`` 中筛选出所有模型参数变量，并根据目录 ``dirname``  或 ``filename`` 提供的参数文件对这些模型参数进行赋值。

使用 ``dirname`` 指定模型参数的存储路径。若模型参数变量以分离文件的形式存储在 ``dirname`` 指定的目录下，则设置 ``filename`` 值为None；若所有模型参数存储在一个单独的二进制文件中，则使用 ``filename`` 来指明这个二进制文件。

注意：
  - 有些变量不是参数，如学习率、全局训练步数（global step）等，但它们之于训练却是必要的。因此，调用 :ref:`cn_api_fluid_io_save_params` 和 :ref:`cn_api_fluid_io_load_params` 来保存和加载参数对于断点训练是不够的，这种情况下可以使用 :ref:`cn_api_fluid_io_save_persistables` 和 :ref:`cn_api_fluid_io_load_persistables` 来保存和加载训练过程的检查点（checkpoint）。
  - 若希望同时加载预训练后的模型结构和模型参数以用于预测过程，则可使用 :ref:`cn_api_fluid_io_load_inference_model` 接口。更多细节请参考 :ref:`api_guide_model_save_reader` 。

参数
::::::::::::

    - **executor**  (Executor) – 加载模型参数的 ``executor`` （详见 :ref:`api_guide_executor` ） 。
    - **dirname**  (str) – 模型参数的存储路径。
    - **main_program**  (Program，可选) – 筛选模型参数变量所依据的 ``Program`` （详见 :ref:`api_guide_Program` ）。若为None，则使用全局默认的  ``default_main_program``。默认值为None。
    - **filename**  (str，可选) – 若模型参数是以若干文件形式存储在 ``dirname`` 指定的目录下，则设置 ``filename`` 值为None。反之，需要通过 ``filename`` 来指明单一模型参数存储文件的名称。默认值为None。

**返回：** 无

代码示例
::::::::::::

COPY-FROM: paddle.fluid.io.load_params