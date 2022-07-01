.. _cn_api_fluid_layers_beam_search_decode:

beam_search_decode
-------------------------------

.. py:function:: paddle.fluid.layers.beam_search_decode(ids, scores, beam_size, end_id, name=None)




该OP用在整个束搜索(Beam search)结束后，通过沿 ``ids`` 中保存的搜索路径回溯，为每个源句（样本）构造完整的beam search结果序列并保存在LoDTensor中。LoDTensor的格式和解析方式如下：

::


    若 lod = [[0, 3, 6], [0, 12, 24, 40, 54, 67, 82]]
    从第一层LoD的内容可以得出：包含两个样本，每个样本均对应了3个（等于束的宽度）生成序列
    从第二层LoD的内容可以得出：第一个样本对应的三个序列的长度分别为12, 12, 16，第一个样本对应的三个序列的长度分别为14, 13, 15。


完整用法请参阅下面的使用示例：

    ::

        fluid/tests/book/test_machine_translation.py

参数
::::::::::::

    - **id** (Variable) - 保存了每个时间步选择的id（beam_search OP的输出）的LoDTensorArray。其中每个LoDTensor的数据类型为int64，LoD level为2，LoD中保存了搜索路径信息。
    - **score** (Variable) - 保存了每个时间步选择的id所对应累积得分（beam_search OP的输出）的LoDTensorArray，和 ``id`` 具有相同大小。其中每个LoDTensor要和 ``id`` 中相应LoDTensor具有相同的形状和LoD，表示其对应的累积得分。数据类型为float32。
    - **beam_size** (int) - 指示束搜索中波束的宽度。
    - **end_id** (int) - 指明标识序列结束的id。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 Variable的二元组，包含了完整id序列和对应的累积得分两个LodTensor，数据类型分别为int64和float32，形状相同且均展开为1维，LoD相同且level均为2。根据两层LoD可分别得到每个源句（样本）有多少个生成序列和每个序列有多少个id。

返回类型
::::::::::::
 tuple


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.beam_search_decode