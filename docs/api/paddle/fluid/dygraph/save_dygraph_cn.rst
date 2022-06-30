.. _cn_api_fluid_dygraph_save_dygraph:

save_dygraph
-------------------------------


.. py:function:: paddle.fluid.dygraph.save_dygraph(state_dict, model_path)




该接口将传入的参数或优化器的 ``dict`` 保存到磁盘上。

``state_dict`` 是通过 :ref:`cn_api_fluid_dygraph_Layer` 的 ``state_dict()`` 方法得到的。

注：``model_path`` 不可以是一个目录。

该接口会根据 ``state_dict`` 的内容，自动给 ``model_path`` 添加 ``.pdparams`` 或者 ``.pdopt`` 后缀，
生成 ``model_path + ".pdparams"`` 或者 ``model_path + ".pdopt"`` 文件。

参数
::::::::::::

 - **state_dict**  (dict of Parameters) – 要保存的模型参数的 ``dict`` 。
 - **model_path**  (str) – 保存state_dict的文件前缀。格式为 ``目录名称/文件前缀``。如果文件前缀为空字符串，会引发异常。

返回
::::::::::::
 无
  
代码示例
::::::::::::

COPY-FROM: paddle.fluid.dygraph.save_dygraph