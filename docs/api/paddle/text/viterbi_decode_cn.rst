.. _cn_api_paddle_text_viterbi_decode:

viterbi_decode
-------------------------------
.. py:function:: paddle.text.viterbi_decode(potentials, transition_params, lengths, include_bos_eos_tag=True, name=None)

该层利用输入的发射概率和转移概率进行解码。通过用 Viterbi 算法，动态地寻找隐藏状态最可能的序列，该序列也被称为 Viterbi 路径（Viterbi path），从而得到观察标签 (tags) 序列。

参数
:::::::::

    - **potentials (Tensor)** 发射概率。形状为[batch_size, lengths, num_tags]，数据类型为 float32 或 float64。
    - **transition_params (Tensor)** 转移概率。形状为[num_tags, num_tags]，数据类型为 float32 或 float64。
    - **lengths (Tensor)** 序列真实长度。形状为[batch_size]，数据类型为 int64。
    - **include_bos_eos_tag (bool，可选)** 是否包含前置、后置标签。如果设为 True，**transition_params** 中倒数第一列为前置标签的转移概率，倒数第二列为后置标签的转移概率。默认值为 True。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::

    - **scores (Tensor)** Viterbi 路径的最高得分。形状为[batch_size]，数据类型为 float32 或 float64。
    - **paths (Tensor)** Viterbi 路径。形状为[batch_size, lengths]，数据类型为 int64。

代码示例
:::::::::

COPY-FROM: paddle.text.viterbi_decode
