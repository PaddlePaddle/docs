.. _cn_api_fluid_layers_open_files:

open_files
-------------------------------

.. py:function:: paddle.fluid.layers.open_files(filenames, shapes, lod_levels, dtypes, thread_num=None, buffer_size=None, pass_num=1, is_test=None)

打开文件(Open files)

该函数获取需要读取的文件列表，并返回Reader变量。通过Reader变量，我们可以从给定的文件中获取数据。所有文件必须有名称后缀来表示它们的格式，例如，``*.recordio``。

参数：
    - **filenames** (list)-文件名列表
    - **shape** (list)-元组类型值列表，声明数据维度
    - **lod_levels** (list)-整形值列表，声明数据的lod层级
    - **dtypes** (list)-字符串类型值列表，声明数据类型
    - **thread_num** (None)-用于读文件的线程数。默认：min(len(filenames),cpu_number)
    - **buffer_size** (None)-reader的缓冲区大小。默认：3*thread_num
    - **pass_num** (int)-用于运行的传递数量
    - **is_test** (bool|None)-open_files是否用于测试。如果用于测试，生成的数据顺序和文件顺序一致。反之，无法保证每一epoch之间的数据顺序是一致的

返回：一个Reader变量，通过该变量获取文件数据

返回类型：变量(Variable)

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    reader = fluid.layers.io.open_files(filenames=['./data1.recordio',
                                            './data2.recordio'],
                                    shapes=[(3,224,224), (1,)],
                                    lod_levels=[0, 0],
                                    dtypes=['float32', 'int64'])

    # 通过reader, 可使用''read_file''层获取数据:
    image, label = fluid.layers.io.read_file(reader)









