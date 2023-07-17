.. _api_guide_cluster_train_data_en:

###############################################
Preparing Data Reader for Distributed Training
###############################################

A parallelism distributed training task usually contains multiple training processes. Each training process processes a part of the entire data set. The unique serial number (trainer_id) of the current process and the total number of training processes (trainers) determines which part of the data can be read by the current training process.

Read datasets in distributed training by defining a cluster_reader
-----------------------------------------------------------------

Generally, you can implement a cluster_reader, regarding the number of training processes and the process serial number(i.e. trainer_id) to decide which data to read:

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

In the code above, `trainers` and `trainer_id` are respectively the total number of training processes and the serial number of the current training process, which can be passed to the Python program through environment variables or parameters.

Split training files in advance
----------------------------------

Since `cluster_reader` is still used to read the full set of data, for tasks with more training processes, it will cause waste of IO resources and affect training performance. Another method is to divide the training data into multiple small files, and each process processes a part of the files.
For example, in a Linux system, the training data can be split into multiple small files using the `split <http://man7.org/linux/man-pages/man1/split.1.html>`_ command:

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

After the data is split, you can define a file_dispatcher function that determines which files need to be read based on the number of training processes and the serial number:

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

        my_files = file_dispatcher(files_pattern, trainers, trainer_id)

In the example above, `files_pattern` is a `glob expression <https://docs.python.org/2.7/library/glob.html>`_ of the training file and can generally be represented by a wildcard.
