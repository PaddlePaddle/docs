.. _cn_api_fluid_layers_beam_search_decode:

beam_search_decode
-------------------------------

.. py:function:: paddle.fluid.layers.beam_search_decode(ids, scores, beam_size, end_id, name=None)




该 OP 用在整个束搜索(Beam search)结束后，通过沿 ``ids`` 中保存的搜索路径回溯，为每个源句（样本）构造完整的 beam search 结果序列并保存在 LoDTensor 中。LoDTensor 的格式和解析方式如下：

::


    若 lod = [[0, 3, 6], [0, 12, 24, 40, 54, 67, 82]]
    从第一层 LoD 的内容可以得出：包含两个样本，每个样本均对应了 3 个（等于束的宽度）生成序列
    从第二层 LoD 的内容可以得出：第一个样本对应的三个序列的长度分别为 12, 12, 16，第一个样本对应的三个序列的长度分别为 14, 13, 15。


完整用法请参阅下面的使用示例：

    ::

        fluid/tests/book/test_machine_translation.py

参数
::::::::::::

    - **id** (Variable) - 保存了每个时间步选择的 id（beam_search OP 的输出）的 LoDTensorArray。其中每个 LoDTensor 的数据类型为 int64，LoD level 为 2，LoD 中保存了搜索路径信息。
    - **score** (Variable) - 保存了每个时间步选择的 id 所对应累积得分（beam_search OP 的输出）的 LoDTensorArray，和 ``id`` 具有相同大小。其中每个 LoDTensor 要和 ``id`` 中相应 LoDTensor 具有相同的形状和 LoD，表示其对应的累积得分。数据类型为 float32。
    - **beam_size** (int) - 指示束搜索中波束的宽度。
    - **end_id** (int) - 指明标识序列结束的 id。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 Variable 的二元组，包含了完整 id 序列和对应的累积得分两个 LodTensor，数据类型分别为 int64 和 float32，形状相同且均展开为 1 维，LoD 相同且 level 均为 2。根据两层 LoD 可分别得到每个源句（样本）有多少个生成序列和每个序列有多少个 id。

返回类型
::::::::::::
 tuple


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.beam_search_decode
