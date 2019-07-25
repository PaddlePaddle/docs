.. _cn_api_fluid_layers_beam_search_decode:

beam_search_decode
-------------------------------

.. py:function:: paddle.fluid.layers.beam_search_decode(ids, scores, beam_size, end_id, name=None)

束搜索层（Beam Search Decode Layer）通过回溯LoDTensorArray ids，为每个源语句构建完整假设，LoDTensorArray ``ids`` 的lod可用于恢复束搜索树中的路径。请参阅下面的demo中的束搜索使用示例：

    ::

        fluid/tests/book/test_machine_translation.py

参数:
        - **id** (Variable) - LodTensorArray，包含所有回溯步骤重中所需的ids。
        - **score** (Variable) - LodTensorArra，包含所有回溯步骤对应的score。
        - **beam_size** (int) - 束搜索中波束的宽度。
        - **end_id** (int) - 结束token的id。
        - **name** (str|None) - 该层的名称(可选)。如果设置为None，该层将被自动命名。

返回： LodTensor 对（pair）， 由生成的id序列和相应的score序列组成。两个LodTensor的shape和lod是相同的。lod的level=2，这两个level分别表示每个源句有多少个假设，每个假设有多少个id。

返回类型: 变量（variable）


**代码示例**

.. code-block:: python

       import paddle.fluid as fluid

       # 假设 `ids` 和 `scores` 为 LodTensorArray变量，它们保留了
       # 选择出的所有时间步的id和score
       ids = fluid.layers.create_array(dtype='int64')
       scores = fluid.layers.create_array(dtype='float32')
       finished_ids, finished_scores = fluid.layers.beam_search_decode(
                ids, scores, beam_size=5, end_id=0)









