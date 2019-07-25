.. _cn_api_fluid_optimizer_PipelineOptimizer:

PipelineOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.PipelineOptimizer(optimizer, cut_list=None, place_list=None, concurrency_list=None, queue_size=30, sync_steps=1, start_cpu_core_id=0)

使用流水线模式进行训练。
Program会根据切分列表cut_list进行分割。如果cut_list的长度是k，则整个program（包括反向部分）将被分割为2*k-1个section。 所以place_list和concurrency_list的长度也必须是2*k-1。 

.. note::

    虽然我们在流水线训练模式中采用异步更新的方式来加速，但最终的效果会依赖于每条流水线的训练进程。我们将在未来尝试同步模式。

参数:
    - **optimizer** (Optimizer) - 基础优化器，如SGD
    - **cut_list** (list of Variable list) - main_program的cut变量列表
    - **place_list** (list of Place) - 对应section运行所在的place
    - **concurrency_list** (list of int) - 指定每个section的并发度列表
    - **queue_size** (int) -  每个section都会消费其输入队列(in-scope queue)中的scope，并向输出队列(out-scope queue)产出scope。 此参数的作用就是指定队列的大小。 可选，默认值：30
    - **sync_steps** (int) - 不同显卡之间的同步周期数。可选，默认值：1
    - **start_cpu_core_id** (int) - 指定所使用的第一个CPU核的id。可选，默认值：0

**代码示例**

.. code-block:: python

        import paddle.fluid as fluid
        import paddle.fluid.layers as layers
        x = fluid.layers.data(name='x', shape=[1], dtype='int64', lod_level=0)
        y = fluid.layers.data(name='y', shape=[1], dtype='int64', lod_level=0)
        emb_x = layers.embedding(input=x, param_attr=fluid.ParamAttr(name="embx"), size=[10,2], is_sparse=False)
        emb_y = layers.embedding(input=y, param_attr=fluid.ParamAttr(name="emby",learning_rate=0.9), size=[10,2], is_sparse=False)
        concat = layers.concat([emb_x, emb_y], axis=1)
        fc = layers.fc(input=concat, name="fc", size=1, num_flatten_dims=1, bias_attr=False)
        loss = layers.reduce_mean(fc)
        optimizer = fluid.optimizer.SGD(learning_rate=0.5)
        optimizer = fluid.optimizer.PipelineOptimizer(optimizer,
                cut_list=[[emb_x, emb_y], [loss]],
                place_list=[fluid.CPUPlace(), fluid.CUDAPlace(0), fluid.CPUPlace()],
                concurrency_list=[1, 1, 4],
                queue_size=2,
                sync_steps=1,
                )
        optimizer.minimize(loss)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        filelist = [] # you should set your own filelist, e.g. filelist = ["dataA.txt"]
        dataset = fluid.DatasetFactory().create_dataset("FileInstantDataset")
        dataset.set_use_var([x,y])
        dataset.set_batch_size(batch_size)
        dataset.set_filelist(filelist)
        exe.train_from_dataset(
                    fluid.default_main_program(),
                    dataset,
                    thread=2,
                    debug=False,
                    fetch_list=[],
                    fetch_info=[],
                    print_period=1)




