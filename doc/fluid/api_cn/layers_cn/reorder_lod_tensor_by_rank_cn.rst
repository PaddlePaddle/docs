.. _cn_api_fluid_layers_reorder_lod_tensor_by_rank:

reorder_lod_tensor_by_rank
-------------------------------

.. py:function:: paddle.fluid.layers.reorder_lod_tensor_by_rank(x, rank_table)


函数参数 ``X`` 是由多个序列(sequence)组成的的一个数据批(batch）。``rank_table`` 存储着batch中序列的重新排列规则。
该算子(operator）根据 ``rank_table`` 中提供的规则信息来实现对 ``X`` 的重新排列。


::

  例如:

  假设在 RankTable 中存储的序列索引为 [3,0,2,1]， X 将会被这样被重新排列：
  X 中的第四个序列（即索引为3的序列，后面以此类推）会变成排列后的batch中的第一个，紧接着就是原来batch中的第一个元素，第三个元素，和第二个元素。
  
  简言之，若有原batch：X = [Seq0, Seq1, Seq2, Seq3] 且 RankTable 中的索引为 [3,0,2,1]，那么输出即为 Out = [Seq3, Seq0, Seq2, Seq1] ，它携带着新的LoD信息。
  如果 X 的LoD信息是空的，这表明 X 不是序列型数据。这和由多个定长为1的序列组成的batch是相同的情况。此时，该函数将对 X 中的切片（slice） 在第一轴(axis)上按 rank_table 里的规则加以排列。
  例如，现有 X = [Slice0, Slice1, Slice2, Slice3] ，并且它LoD信息为空，在 RankTable 索引为[3, 0, 2, 1]。则 Out = [Slice3, Slice0, Slice2, Slice1] ，并且不在其中追加LoD信息。
  注意，该operator对 ``X`` 进行的排序所依据的 ``LoDRankTable`` 不一定是在 ``X`` 的基础上得出来的。它可以由其他不同的序列得出，并由该operator依据这个 ``LoDRankTable`` 来对  ``X`` 排序。

参数：
    - **x(Variable)** - (LoDTensor)，待根据提供的 ``RankTable`` 进行排序的LoD tensor
    - **rank_table(Variable)** - 变量


返回： 重新排列后的LoDTensor

返回类型: out(Variable)

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data_desc = (['input', [9], 0], ['ref', [5], 1])
    data = fluid.layers.data(name=data_desc[0][0], shape=data_desc[0][1])
    rank_data = fluid.layers.data(name=data_desc[1][0], shape=data_desc[1][1])
    table = fluid.layers.control_flow.lod_rank_table(rank_data)
    new_data = fluid.layers.reorder_lod_tensor_by_rank(
                     x=data, rank_table=table)










