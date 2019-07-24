.. _cn_api_fluid_layers_Preprocessor:

Preprocessor
-------------------------------

.. py:class:: paddle.fluid.layers.Preprocessor(reader, name=None)

reader变量中数据预处理块。

参数：
    - **reader** (Variable)-reader变量
    - **name** (str,默认None)-reader的名称

**代码示例**:

.. code-block:: python

    import paddle.fluid as fluid
    reader = fluid.layers.io.open_files(
        filenames=['./data1.recordio', './data2.recordio'],
        shapes=[(3, 224, 224), (1, )],
        lod_levels=[0, 0],
        dtypes=['float32', 'int64'])

    preprocessor = fluid.layers.io.Preprocessor(reader=reader)
    with preprocessor.block():
        img, lbl = preprocessor.inputs()
        img_out = img / 2
        lbl_out = lbl + 1
        preprocessor.outputs(img_out, lbl_out)
    data_file = fluid.layers.io.double_buffer(preprocessor())









