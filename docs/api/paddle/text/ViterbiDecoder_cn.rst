.. _cn_api_paddle_text_ViterbiDecoder:

ViterbiDecoder
-------------------------------
.. py:class:: paddle.text.ViterbiDecoder(transitions, include_bos_eos_tag=True, name=None)

构建一个 ``ViterbiDecoder`` 类的可调用对象。请参见 :ref:`cn_api_paddle_text_viterbi_decode` API。

参数
:::::::::
    - **transitions (Tensor)** 转移概率。形状为[num_tags, num_tags]，数据类型为 float32 或 float64。
    - **include_bos_eos_tag (bool，可选)** 是否包含前置、后置标签。如果设为 True，**transition_params** 中倒数第一列为前置标签的转移概率，倒数第二列为后置标签的转移概率。默认值为 True。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
:::::::::
    - **potentials (Tensor)** 发射概率。形状为[batch_size, sequence_length, num_tags]，数据类型为 float32 或 float64。
    - **lengths (Tensor)** 序列真实长度。形状为[batch_size]，数据类型为 int64。

返回
:::::::::
    - **scores (Tensor)** Viterbi 路径的最高得分。形状为[batch_size]，数据类型为 float32 或 float64。
    - **paths (Tensor)** Viterbi 路径。形状为[batch_size, sequence_length]，数据类型为 int64。

代码示例
:::::::::

COPY-FROM: paddle.text.ViterbiDecoder
