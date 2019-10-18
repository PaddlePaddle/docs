.. _cn_api_fluid_dygraph_load_dygraph:

load_dygraph
-------------------------------

.. py:function:: paddle.fluid.dygraph.load_dygraph(model_path)

该接口尝试从磁盘中加载参数或优化器的dict。

参数:
    - **model_path**  (str) – 保存state_dict的文件前缀。该路径不应该包括后缀 ``.pdparams`` 或 ``.pdopt``。


返回: 两个dict，即从文件中恢复的参数dict和优化器dict

- para_dict: 从文件中恢复的参数dict
- opti_dict: 从文件中恢复的优化器dict

返回类型: tuple(dict, dict)
  
**代码示例**

.. code-block:: python

    import paddle.fluid as fluid

    prog = fluid.default_main_program()
    fluid.save( prog, "./temp")
    fluid.load( prog, "./temp")



