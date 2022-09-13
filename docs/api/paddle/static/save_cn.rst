.. _cn_api_fluid_save:

save
-------------------------------

.. py:function:: paddle.static.save(program, model_path, protocol=4, **configs)


将传入的参数、优化器信息和网络描述保存到 ``model_path`` 。

参数包含所有的可训练 :ref:`cn_api_fluid_Variable`，将保存到后缀为 ``.pdparams`` 的文件中。

优化器信息包含优化器使用的所有 Tensor。对于 Adam 优化器，包含 beta1、beta2、momentum 等。
所有信息将保存到后缀为 ``.pdopt`` 的文件中。（如果优化器没有需要保存的 Tensor（如 sgd），则不会生成）。

网络描述是程序的描述。它只用于部署。描述将保存到后缀为 ``.pdmodel`` 的文件中。

参数
::::::::::::

 - **program**  ( :ref:`cn_api_fluid_Program` ) – 要保存的 Program。
 - **model_path**  (str) – 保存 program 的文件前缀。格式为 ``目录名称/文件前缀``。如果文件前缀为空字符串，会引发异常。
 - **protocol**  (int，可选) – pickle 模块的协议版本，默认值为 4，取值范围是[2,4]。
 - **\*\*configs**  (dict，可选) - 可选的关键字参数。

返回
::::::::::::

无。

代码示例
::::::::::::

COPY-FROM: paddle.static.save
