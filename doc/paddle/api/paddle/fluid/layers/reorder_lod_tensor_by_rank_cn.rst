.. _cn_api_fluid_layers_reorder_lod_tensor_by_rank:

reorder_lod_tensor_by_rank
-------------------------------

.. py:function:: paddle.fluid.layers.reorder_lod_tensor_by_rank(x, rank_table)





该OP根据 ``rank_table`` 中提供的 ``LoDRankTable`` 类型的顺序信息来实现对 ``X`` 的重新排列。
接口参数 ``X`` 是由多个序列(Sequence)组成的的一个批序列（Batch of Sequences）， ``rank_table`` 存储着对batch中序列重新排列的 ``LoDRankTable`` 类型的顺序信息。

例如:

假设在 ``rank_table`` 中存储的序列索引为 :math:`[3,0,2,1]` ， ``X``  将会被这样被重新排列：
``X`` 中的第四个序列（即索引为3的序列，后面以此类推）会变成排列后的batch中的第一个，紧接着就是原来batch中的第一个元素，第三个元素，和第二个元素。
简言之，若有原batch：:math:`X = [Seq0, Seq1, Seq2, Seq3]` 且 RankTable 中的索引为 :math:`[3,0,2,1]` ，那么输出即为 :math:`Out = [Seq3, Seq0, Seq2, Seq1]` ，它携带着新的LoD信息。
如果 ``X`` 的LoD信息是空的，这表明 ``X`` 不是序列型数据。这和由多个定长为1的序列组成的batch是相同的情况。此时，该函数将对 ``X`` 中数据 在第一轴(axis)上按 ``rank_table`` 里的规则加以排列。
例如，现有 :math:`X = [Slice0, Slice1, Slice2, Slice3]` ，并且它LoD信息为空，在 ``rank_table`` 索引为 :math:`[3, 0, 2, 1]` 。则 :math:`Out = [Slice3, Slice0, Slice2, Slice1]` ，并且不在其中追加LoD信息。

注意：该OP对 ``X`` 进行的排序所依据的 ``LoDRankTable`` 不一定是在 ``X`` 的基础上得出来的。它可以由其他不同的序列得出，并由该OP依据这个 ``LoDRankTable`` 来对 ``X`` 排序。

参数：
    - **x** (Variable) - 待根据提供的 ``rank_table`` 进行排序的LoDTensor.
    - **rank_table** (Variable) - 提供对 ``x`` 重新排列的 ``LoDRankTable`` 类型的顺序信息.


返回： 重新排列后的LoDTensor

返回类型: Variable

**代码示例**：

.. code-block:: python

    
    import numpy as np
    import paddle.fluid as fluid

    rank_data = fluid.layers.data(name='rank_data', shape=[5], dtype='float32', lod_level=2)
    table = fluid.layers.control_flow.lod_rank_table(rank_data, level=1)

    data = fluid.layers.data(name='data', shape=[9], lod_level=2)
    new_data = fluid.layers.reorder_lod_tensor_by_rank(
                     x=data, rank_table=table)


    place=fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    rank_tensor = fluid.create_lod_tensor(np.random.random([14,5]).astype("float32"), [[4,1], [3, 2, 2, 3, 4]], place)

    data_ndarray = np.random.random([27, 9]).astype("float32")
    data_lod = [[1, 2, 2, 4, 4], [2, 2, 4, 2, 2, 2, 1, 1, 2, 2, 4, 2, 1]]
    data_tensor = fluid.create_lod_tensor(data_ndarray, data_lod, place)

    out = exe.run(fluid.default_main_program(),feed={'data':data_tensor, 'rank_data':rank_tensor}, fetch_list=[new_data], return_numpy=False)
    print(out[0])
    # lod: {{0, 4, 5, 9, 11, 13}{0, 2, 6, 8, 9, 11, 13, 14, 15, 17, 19, 23, 25, 27}}
    #shape: [27, 9]




