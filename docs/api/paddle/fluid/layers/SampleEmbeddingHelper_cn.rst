.. _cn_api_fluid_layers_SampleEmbeddingHelper:

SampleEmbeddingHelper
-------------------------------


.. py:class:: paddle.fluid.layers.SampleEmbeddingHelper(embedding_fn, start_tokens, end_token, softmax_temperature=None, seed=None)

SampleEmbeddingHelper是 :ref:`cn_api_fluid_layers_GreedyEmbeddingHelper` 的子类。作为解码helper，它通过采样而非使用 :code:`argmax` 并将采样结果送入embedding层，以此作为下一解码步的输入。

参数
::::::::::::

  - **embedding_fn** (callable) - 作用于 :code:`argmax` 结果的函数，通常是一个将词id转换为词嵌入的embedding层，**注意**，这里要使用 :ref:`cn_api_fluid_embedding` 而非 :ref:`cn_api_fluid_layers_embedding`，因为选中的id的形状是 :math:`[batch\_size]`，如果使用后者则还需要在这里提供unsqueeze。
  - **start_tokens** (Variable) - 形状为 :math:`[batch\_size]` 、数据类型为int64、 值为起始标记id的tensor。
  - **end_token** (int) - 结束标记id。
  - **softmax_temperature** (float，可选) - 该值用于在softmax计算前除以logits。温度越高（大于1.0）随机性越大，温度越低则越趋向于argmax。该值必须大于0，默认值None等同于1.0。
  - **seed** (int，可选) - 采样使用的随机种子。默认为None，表示不使用固定的随机种子。

代码示例
::::::::::::


COPY-FROM: paddle.fluid.layers.SampleEmbeddingHelper

方法
::::::::::::
sample(time, outputs, states)
'''''''''

根据一个多项分布进行采样，此分布由 :code:`softmax(outputs/softmax_temperature)` 计算得到。

**参数**

  - **time** (Variable) - 调用者提供的形状为[1]的tensor，表示当前解码的时间步长。其数据类型为int64。
  - **outputs** (Variable) - tensor变量，通常其数据类型为float32或float64，形状为 :math:`[batch\_size, vocabulary\_size]`，表示当前解码步预测产生的logit（未归一化的概率），和由 :code:`BasicDecoder.output_fn(BasicDecoder.cell.call())` 返回的 :code:`outputs` 是同一内容。
  - **states** (Variable) - 单个tensor变量或tensor变量组成的嵌套结构，和由 :code:`BasicDecoder.cell.call()` 返回的 :code:`new_states` 是同一内容。

**返回**
数据类型为int64形状为 :math:`[batch\_size]` 的tensor，表示采样得到的id。

**返回类型**
Variable
