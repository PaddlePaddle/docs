.. _cn_api_fluid_io_load_inference_model:

load_inference_model
-------------------------------


.. py:function:: paddle.static.load_inference_model(path_prefix, executor, **kwargs)




从指定文件路径中加载预测模型，包括模型结构和模型参数。

参数
::::::::::::

  - **path_prefix** (str) – 模型的存储目录 + 模型名称（不包含后缀）。如果是 None，表示从内存加载模型。
  - **executor** (Executor) – 运行模型的 ``executor``，详见 :ref:`api_guide_executor` 。
  - **kwargs** - 支持的 key 包括 'model_filename', 'params_filename'。(注意：kwargs 主要是用来做反向兼容的)。

      - **model_filename** (str) - 自定义 model_filename。

      - **params_filename** (str) - 自定义 params_filename。

返回
::::::::::::
该接口返回一个包含三个元素的列表 [program，feed_target_names, fetch_targets]。它们的含义描述如下：

  - **program** （Program）– ``Program`` （详见 :ref:`api_guide_Program` ）类的实例。此处它被用于预测，因此可被称为 Inference Program。
  - **feed_target_names** （list）– 字符串列表，包含着 Inference Program 预测时所需提供数据的所有变量名称（即所有输入变量的名称）。
  - **fetch_targets** （list）– ``Variable`` （详见 :ref:`api_guide_Program` ）类型列表，包含着模型的所有输出变量。通过这些输出变量即可得到模型的预测结果。


代码示例
::::::::::::

COPY-FROM: paddle.static.load_inference_model
