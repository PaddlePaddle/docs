..  _api_guide_model_save_reader:

#########
模型保存与加载
#########

模型的保存与加载主要涉及到如下八个 API：
:code:`fluid.io.save_vars`、:code:`fluid.io.save_params`、:code:`fluid.io.save_persistables`、:code:`fluid.io.save_inference_model`、:code:`fluid.io.load_vars`、:code:`fluid.io.load_params`、:code:`fluid.io.load_persistables` 和 :code:`fluid.io.load_inference_model`。

变量、持久性变量和参数
====================

在 :code:`Paddle` 中，算子(:code:`Operator`)的每一个输入和输出都是一个变量（:code:`Variable`），而参数（:code:`Parameter`）是变量（:code:`Variable`）的子类。持久性变量（:code:`Persistables`）是一种在每次迭代结束后均不会被删除的变量。参数是一种持久性变量，其在每次迭代后都会被优化器（:ref:`api_guide_optimizer`）更新。训练神经网络本质上就是在更新参数。

模型保存 API 介绍
====================

- :code:`fluid.io.save_vars`：通过执行器（:ref:`api_guide_executor`）保存变量到指定的目录中。保存变量的方式有两种：

  1）通过接口中的 :code:`vars` 指定需要保存的变量列表。

  2）将一个已经存在的程序（:code:`Program`）赋值给接口中的 :code:`main_program`，然后这个程序中的所有变量都将被保存下来。

  第一种保存方式的优先级要高于第二种。

  API Reference 请参考 :ref:`cn_api_fluid_io_save_vars`。

- :code:`fluid.io.save_params`：通过接口中的 :code:`main_program` 指定好程序（:code:`Program`），该接口会将所指定程序中的全部参数（:code:`Parameter`）过滤出来，并将它们保存到 :code:`dirname` 指定的文件夹或 :code:`filename` 指定的文件中。

  API Reference 请参考 :ref:`cn_api_fluid_io_save_params`。

- :code:`fluid.io.save_persistables`：通过接口中的 :code:`main_program` 指定好程序（:code:`Program`），该接口会将所指定程序中的全部持久性变量（:code:`persistable==True`）过滤出来，并将它们保存到 :code:`dirname` 指定的文件夹或 :code:`filename` 指定的文件中。

  API Reference 请参考 :ref:`cn_api_fluid_io_save_persistables`。

- :code:`fluid.io.save_inference_model`：请参考  :ref:`api_guide_inference`。

模型加载 API 介绍
====================

- :code:`fluid.io.load_vars`：通过执行器（:code:`Executor`）加载指定目录中的变量。加载变量的方式有两种：

  1）通过接口中的 :code:`vars` 指定需要加载的变量列表。

  2）将一个已经存在的程序（:code:`Program`）赋值给接口中的 :code:`main_program`，然后这个程序中的所有变量都将被加载。

  第一种加载方式的优先级要高于第二种。

  API Reference 请参考 :ref:`cn_api_fluid_io_load_vars`。

- :code:`fluid.io.load_params`：该接口从 :code:`main_program` 指定的程序中过滤出全部参数（:code:`Parameter`），并试图从 :code:`dirname` 指定的文件夹或 :code:`filename` 指定的文件中加载这些参数。

  API Reference 请参考 :ref:`cn_api_fluid_io_load_params`。

- :code:`fluid.io.load_persistables`：该接口从 :code:`main_program` 指定的程序中过滤出全部持久性变量（:code:`persistable==True`），并试图从 :code:`dirname` 指定的文件夹或 :code:`filename` 指定的文件中加载这些持久性变量。

  API Reference 请参考 :ref:`cn_api_fluid_io_load_persistables`。

-  :code:`fluid.io.load_inference_model`：请参考  :ref:`api_guide_inference`。
