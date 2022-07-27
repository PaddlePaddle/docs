..  _api_guide_cluster_train_data:

####################
分布式训练 reader 准备
####################

一个数据并行的分布式训练任务通常会含有多个训练进程，每个训练进程处理整个数据集中的一部分，根据当前进程的唯一序号(trainer_id)以及训练进程总数(trainers)可以决定当前训练进程应该读取哪一部分数据。

实现 cluster_reader 来读取分布式训练数据集
----------------------------------------

比较通用的方法，可以实现一个 cluster_reader, 根据训练进程数量以及进程序号决定读取哪些 example:

    .. code-block:: python

        def cluster_reader(reader, trainers, trainer_id):
            def reader_creator():
                for idx, data in enumerate(reader()):
                    if idx % trainers == trainer_id:
                        yield data
            return reader

        trainers = int(os.getenv("PADDLE_TRAINERS", "1"))
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        train_reader = cluster_reader(paddle.dataset.mnist.train(), trainers, trainer_id)

上述代码中，`trainers` 和 `trainer_id` 分别是训练进程总数和当前训练进程的序号，可以通过环境变量或者参数的方式传递给 Python 程序。

预先切分训练文件
-----------------

由于使用 `cluster_reader` 依然会读取全量数据，对于训练进程比较多的任务，会造成 IO 资源的浪费、影响训练性能。另一种方法是可以将训练数据切分成多个小文件，每个进程处理其中的一部分文件,
例如在 Linux 系统中可以使用 `split <http://man7.org/linux/man-pages/man1/split.1.html>`_ 命令将训练数据切分成多个小文件：

  .. code-block:: bash
    $ split -d -a 4 -d -l 100 housing.data cluster/housing.data.
    $ find ./cluster
    cluster/
    cluster/housing.data.0002
    cluster/housing.data.0003
    cluster/housing.data.0004
    cluster/housing.data.0000
    cluster/housing.data.0001
    cluster/housing.data.0005

数据切分好以后, 可以实现一个 file_dispatcher 函数，根据训练进程数量以及序号决定需要读取哪些文件：

    .. code-block:: python

        def file_dispatcher(files_pattern, trainers, trainer_id):
            file_list = glob.glob(files_pattern)
            ret_list = []
            for idx, f in enumerate(file_list):
                if (idx + trainers) % trainers == trainer_id:
                    ret_list.append(f)
            return ret_list

        trainers = int(os.getenv("PADDLE_TRAINERS", "1"))
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        files_pattern = "cluster/housing.data.*"

        my_files = file_dispatcher(files_pattern, triners, trainer_id)

在上述例子中，`files_pattern` 是训练文件的 `glob 表达式 <https://docs.python.org/2.7/library/glob.html>`_，一般可以用通配符来表示。
