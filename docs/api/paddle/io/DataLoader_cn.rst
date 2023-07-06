.. _cn_api_fluid_io_DataLoader:

DataLoader
-------------------------------

.. py:class:: paddle.io.DataLoader(dataset, feed_list=None, places=None, return_list=False, batch_sampler=None, batch_size=1, shuffle=False, drop_last=False, collate_fn=None, num_workers=0, use_buffer_reader=True, use_shared_memory=True, prefetch_factor=2, timeout=0, worker_init_fn=None)

DataLoader 返回一个迭代器，该迭代器根据 ``batch_sampler`` 给定的顺序迭代一次给定的 ``dataset``

DataLoader 支持单进程和多进程的数据加载方式，当 ``num_workers`` 大于 0 时，将使用多进程方式异步加载数据。

DataLoader 当前支持 ``map-style`` 和 ``iterable-style`` 的数据集，``map-style`` 的数据集可通过下标索引样本，请参考 ``paddle.io.Dataset`` ； ``iterable-style`` 数据集只能迭代式地获取样本，类似 Python 迭代器，请参考 ``paddle.io.IterableDataset`` 。

.. note::

    当前还不支持在子进程中进行 GPU Tensor 的操作，请不要在子进程流程中使用 GPU Tensor，例如 ``dataset`` 中的预处理，``collate_fn`` 等，``numpy array`` 和 CPU Tensor 操作已支持。

``batch_sampler`` 请参考 ``paddle.io.BatchSampler``

**禁用自动组 batch**

在如 NLP 等任务中，用户需求自定义组 batch 的方式，不希望 ``DataLoader`` 自动组 batch， ``DataLoader`` 支持在 ``batch_size`` 和 ``batch_sampler`` 均为 None 的时候禁用自动组 batch 功能，此时需求从 ``dataset`` 中获取的数据为已经组好 batch 的数据，该数据将不做任何处理直接传到 ``collate_fn`` 或 ``default_collate_fn`` 中。

.. note::

    当禁用自动组 batch 时，``default_collate_fn`` 将不对输入数据做任何处理。

参数
::::::::::::

    - **dataset** (Dataset) - DataLoader 从此参数给定数据集中加载数据，此参数必须是 ``paddle.io.Dataset`` 或 ``paddle.io.IterableDataset`` 的一个子类实例。
    - **feed_list** (list(Tensor)|tuple(Tensor)，可选) - feed 变量列表，由 ``paddle.static.data()`` 创建。当 ``return_list`` 为 False 时，此参数必须设置。默认值为 None。
    - **places** (list(Place)|tuple(Place)，可选) - 数据需要放置到的 Place 列表。在静态图和动态图模式中，此参数均必须设置。在动态图模式中，此参数列表长度必须是 1。默认值为 None。
    - **return_list** (bool，可选) - 每个设备上的数据是否以 list 形式返回。若 return_list = False，每个设备上的返回数据均是 str -> Tensor 的映射表，其中映射表的 key 是每个输入变量的名称。若 return_list = True，则每个设备上的返回数据均是 list(Tensor)。在动态图模式下，此参数必须为 True。默认值为 False。
    - **batch_sampler** (BatchSampler，可选) - ``paddle.io.BatchSampler`` 或其子类的实例，DataLoader 通过 ``batch_sampler`` 产生的 mini-batch 索引列表来 ``dataset`` 中索引样本并组成 mini-batch。默认值为 None。
    - **batch_size** (int|None，可选) - 每 mini-batch 中样本个数，为 ``batch_sampler`` 的替代参数，若 ``batch_sampler`` 未设置，会根据 ``batch_size`` ``shuffle`` ``drop_last`` 创建一个 ``paddle.io.BatchSampler``。默认值为 1。
    - **shuffle** (bool，可选) - 生成 mini-batch 索引列表时是否对索引打乱顺序，为 ``batch_sampler`` 的替代参数，若 ``batch_sampler`` 未设置，会根据 ``batch_size`` ``shuffle`` ``drop_last`` 创建一个 ``paddle.io.BatchSampler``。默认值为 False。
    - **drop_last** (bool，可选) - 是否丢弃因数据集样本数不能被 ``batch_size`` 整除而产生的最后一个不完整的 mini-batch，为 ``batch_sampler`` 的替代参数，若 ``batch_sampler`` 未设置，会根据 ``batch_size`` ``shuffle`` ``drop_last`` 创建一个 ``paddle.io.BatchSampler``。默认值为 False。
    - **collate_fn** (callable，可选) - 通过此参数指定如果将样本列表组合为 mini-batch 数据，当 ``collate_fn`` 为 None 时，默认为将样本个字段在第 0 维上堆叠(同 ``np.stack(..., axis=0)`` )为 mini-batch 的数据。默认值为 None。
    - **num_workers** (int，可选) - 用于加载数据的子进程个数，若为 0 即为不开启子进程，在主进程中进行数据加载。默认值为 0。
    - **use_buffer_reader** (bool，可选) - 是否使用缓存读取器。若 ``use_buffer_reader`` 为 True，DataLoader 会异步地预读取一定数量（默认读取下一个）的 mini-batch 的数据，可加速数据读取过程，但同时会占用少量的 CPU/GPU 存储，即一个 batch 输入数据的存储空间。默认值为 True。
    - **prefetch_factor** (int，可选) - 缓存的 mini-batch 的个数。若 ``use_buffer_reader`` 为 True，DataLoader 会异步地预读取 ``prefetch_factor`` 个 mini-batch。默认值为 2。
    - **use_shared_memory** (bool，可选) - 是否使用共享内存来提升子进程将数据放入进程间队列的速度，该参数尽在多进程模式下有效(即 ``num_workers > 0`` )，请确认机器上有足够的共享内存空间(如 Linux 系统下 ``/dev/shm/`` 目录空间大小)再设置此参数。默认为 True。
    - **timeout** (int，可选) - 从子进程输出队列获取 mini-batch 数据的超时时间。默认值为 0。
    - **worker_init_fn** (callable，可选) - 子进程初始化函数，此函数会被子进程初始化时被调用，并传递 ``worker id`` 作为参数。默认值为 None。

返回
::::::::::::
DataLoader，迭代 ``dataset`` 数据的迭代器，迭代器返回的数据中的每个元素都是一个 Tensor。


代码示例
::::::::::::

COPY-FROM: paddle.io.DataLoader:data-loader-example

方法
::::::::::::
from_generator(feed_list=None, capacity=None, use_double_buffer=True, iterable=True, return_list=False, use_multiprocess=False, drop_last=True)
'''''''''

.. warning::
    这个 API 将在未来版本废弃，推荐使用支持多进程并发加速的 ``paddle.io.DataLoader``

.. note::
    框架保证 DataLoader 的数据加载顺序与用户提供的数据源读取顺序一致。

创建一个 DataLoader 对象用于加载 Python 生成器产生的数据。数据会由 Python 线程预先读取，并异步送入一个队列中。

本方法创建的 DataLoader 对象提供了 3 个方法设置数据源，分别是 :code:`set_sample_generator` , :code:`set_sample_list_generator` 和
:code:`set_batch_generator`。请查阅下述示例代码了解它们的使用方法。

如果 iterable = True，本方法创建的 DataLoader 对象是一个 Python 生成器，可以 for-range 的方法循环迭代。

如果 iterable = False，本方法创建的 DataLoader 对象提供 :code:`start()` 和 :code:`reset()` 方法控制数据读取过程。

**参数**

    - **feed_list** (list(Tensor)|tuple(Tensor)) - feed 变量列表，由 ``paddle.static.data()`` 创建。
    - **capacity** (int) - DataLoader 对象内部维护队列的容量大小。单位是 batch 数量。若 reader 读取速度较快，建议设置较大的 capacity 值。
    - **use_double_buffer** (bool，可选) - 是否使用 ``double_buffer_reader``。若 use_double_buffer=True，DataLoader 会异步地预读取下一个 batch 的数据，可加速数据读取过程，但同时会占用少量的 CPU/GPU 存储，即一个 batch 输入数据的存储空间。
    - **iterable** (bool，可选) - 所创建的 DataLoader 对象是否可迭代。
    - **return_list** (bool，可选) - 每个设备上的数据是否以 list 形式返回。仅在 iterable = True 模式下有效。若 return_list = False，每个设备上的返回数据均是 str -> Tensor 的映射表，其中映射表的 key 是每个输入变量的名称。若 return_list = True，则每个设备上的返回数据均是 list(Tensor)。推荐在静态图模式下使用 return_list = False，在动态图模式下使用 return_list = True。
    - **use_multiprocess** (bool，可选) - 设置是否是用多进程加速动态图的数据载入过程。注意：该参数的设置仅在动态图模式下有效，在静态图模式下，该参数设置与否均无任何影响。默认值为 False。
    - **drop_last** (bool，可选)：是否丢弃最后的不足 CPU/GPU 设备数的批次。默认值为 True。在网络训练时，用户不能设置 drop_last=False，此时所有 CPU/GPU 设备均应从 DataLoader 中读取到数据。在网络预测时，用户可以设置 drop_last=False，此时最后不足 CPU/GPU 设备数的批次可以进行预测。

**返回**

 被创建的 DataLoader 对象。


**代码示例 1**

COPY-FROM: paddle.fluid.DataLoader.from_generator:static-data-loader-example-1

**代码示例 2**

COPY-FROM: paddle.fluid.DataLoader.from_generator:static-data-loader-example-2

**代码示例 3**

.. code-block:: python

    '''
    Example of `drop_last` using in static graph multi-cards mode
    '''
    import paddle
    import paddle.static as static
    import numpy as np
    import os

    # We use 2 CPU cores to run inference network
    os.environ['CPU_NUM'] = '2'

    paddle.enable_static()

    # The data source has only 3 batches, which can not be
    # divided evenly to each CPU core
    def batch_generator():
        for i in range(3):
            yield np.array([i+1]).astype('float32'),

    x = static.data(name='x', shape=[None], dtype='float32')
    y = x * x

    def run_inference(drop_last):
        loader = paddle.io.DataLoader.from_generator(feed_list=[x],
                capacity=8, drop_last=drop_last)
        loader.set_batch_generator(batch_generator, static.cpu_places())

        exe = static.Executor(paddle.CPUPlace())
        prog = static.CompiledProgram(static.default_main_program())

        result = []
        for data in loader():
            each_ret, = exe.run(prog, feed=data, fetch_list=[y])
            result.extend(each_ret)
        return result

    # Set drop_last to True, so that the last batch whose
    # number is less than CPU core number would be discarded.
    print(run_inference(drop_last=True)) # [1.0, 4.0]

    # Set drop_last to False, so that the last batch whose
    # number is less than CPU core number can be tested.
    print(run_inference(drop_last=False)) # [1.0, 4.0, 9.0]


from_dataset(dataset, places, drop_last=True)
'''''''''

.. warning::
    这个 API 将在未来版本废弃，推荐使用支持多进程并发加速的 ``paddle.io.DataLoader``

创建一个 DataLoader 对象用于加载 Dataset 产生的数据。目前，Dataset 仅支持 Linux 系统下使用。

**参数**

    - **dataset** (InMemoryDataset|QueueDataset) - Dataset 对象。
    - **places** (list(CUDAPlace)|list(CPUPlace)) - DataLoader 对象返回数据所在的 place。
    - **drop_last** (bool，可选) - 是否丢弃最后样本数量不足 batch size 的 batch。若 drop_last = True 则丢弃，若 drop_last = False 则不丢弃。

**返回**

 被创建的 DataLoader 对象，可以 for-range 的方式循环迭代。


**代码示例**

COPY-FROM: paddle.fluid.DataLoader.from_dataset
