
..  _cluster_example_cpups:

CPUPS 流式训练示例
-------------------------

在之前的参数服务器概述中曾经提到，由于推荐搜索场景的特殊性，在训练过程中使用到的训练数据通常不是固定的集合，而是随时间流式的加入到训练过程中，这种训练方式称为流式训练。

与传统的固定数据集的训练相比，在大规模稀疏参数场景下，流式训练有如下特点：

1. 在线上系统提供服务的过程中，训练程序一直运行，等待线上服务产生的数据，并在数据准备完成后进行增量训练。
2. 数据一般按时间组织，每个 Pass 的训练数据对应一个或几个时间分片（文件夹），并在该时间分片的数据生成结束后，在对应时间分片的文件夹中添加一个空文件用于表示数据准备完成。
3. 线上服务产生的数据不断进入训练系统，导致稀疏参数不断增加，在模型精度不受影响的情况下，为控制模型总存储，增加稀疏参数统计量自动统计、稀疏参数准入、退场、增量保存等功能。

在学习流式训练具体使用方法之前，建议先详细阅读\ `参数服务器快速开始 <../cluster_quick_start_ps_cn.html>`_\章节，了解参数服务器的基本使用方法。

流式训练的完整代码示例参见：\ `PaddleRec <https://github.com/PaddlePaddle/PaddleRec>`_\，具体的目录结构如下：

.. code-block:: text

    ├── tools
        ├── static_ps_online_trainer.py      # 流式训练主脚本
        ├── utils                            # 流式训练所需各种工具函数封装
            ├── static_ps
                ├── flow_helper.py           # 流式训练所需 utils，包括保存、加载等
                ├── metric_helper.py         # 分布式指标计算所需 utils
                ├── time_helper.py           # 训练耗时计算所需 utils
                ├── program_helper.py        # 模型引入、分布式 strategy 生成所需 util
    ├── models                               # 存放具体模型组网和配置
        ├── rank
            ├── slot_dnn                     # 模型示例目录
                ├── net.py                   # mlp 具体组网
                ├── static_model.py          # 静态图训练调用（在 net.py 基础上封装）
                ├── config_online.yaml       # 流式训练配置文件
                ├── queuedataset_reader.py   # 数据处理脚本

1 模型组网
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

为实现稀疏参数统计量自动统计，在组网时需要注意以下两点：

1. embedding 层需使用 ``paddle.static.nn.sparse_embedding`` ，该算子为大规模稀疏参数模型特定的 embedding 算子，支持对稀疏参数统计量自动统计。
2. 稀疏参数的统计量，目前特指稀疏特征的展现量(show)和点击量(click)，在组网时定义两个变量，指明特征是否展现和点击，通常取值为 0 或者 1， ``sparse_embedding`` 中通过 entry 参数传入一个 ``ShowClickEntry`` ，指明这两个变量(show 和 click)的名字。

.. code-block:: python

    # static_model.py
    # 构造 show/click 对应的 data，变量名需要与 entry 中的名称一致
    show = paddle.static.data(
        name="show", shape=[None, 1], dtype="int64")
    label = paddle.static.data(
        name="click", shape=[None, 1], dtype="int64")

    # net.py
    # ShowClickEntry 接收的 show/click 变量数据类型为 float32，需做 cast 处理
    show_cast = paddle.cast(show, dtype='float32')
    click_cast = paddle.cast(click, dtype='float32')

    # 构造 ShowClickEntry，指明展现和点击对应的变量名
    self.entry = paddle.distributed.ShowClickEntry(show_cast.name,
                                                   click_cast.name)
    emb = paddle.static.nn.sparse_embedding(
        input=s_input,
        size=[self.dict_dim, self.emb_dim],
        padding_idx=0,
        entry=self.entry,   # 在 sparse_embedding 中传入 entry
        param_attr=paddle.ParamAttr(name="embedding"))

2 数据读取
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

本章节主要讲解流式训练中的数据组织形式、数据等待方式、数据拆分方式等内容，涉及到的概念如下：

1. 数据分片：将数据按照固定的时间间隔组织成多个分片，一段时间间隔中的所有训练数据称为一个数据分片。
2. Pass：把当前训练集的所有数据全部训练一遍，通常一个 Pass 对应一个或多个数据分片。

涉及到的具体配置如下：

.. csv-table::
    :header: "名称", "类型", "取值", "是否必须", "作用描述"
    :widths: 10, 5, 5, 5, 30

    "train_data_dir", "string", "任意", "是", "训练数据所在目录（如果数据存于 afs 或 hdfs 上，以 afs:/或 hdfs:/开头）"
    "split_interval", "int", "任意", "是", "数据落盘分片间隔时间（分钟）"
    "split_per_pass", "int", "任意", "是", "训练一个 Pass 包含多少个分片的数据"
    "start_day", "string", "任意", "是", "训练开始的日期（例：20190720）"
    "end_day", "string", "任意", "是", "训练结束的日期（例：20190720）"
    "data_donefile", "string", "任意", "否", "用于探测当前分片数据是否落盘完成的标识文件"
    "data_sleep_second", "int", "任意", "否", "当前分片数据尚未完成的等待时间"
    "prefetch", "bool", "任意", "否", "是否开启数据 prefetch，即在当前 pass 训练过程中预读取下一个 pass 的数据"

2.1 数据组织形式
""""""""""""

在训练数据目录下，再建立两层目录，第一层目录对应训练数据的日期（8 位），第二层目录对应训练数据的具体时间（4 位，前两位为小时，后两位为分钟），并且需要与配置文件中的 split_interval 配置对应。
例如：train_data_dir 配置为“data”目录，split_interval 配置为 5，则具体的目录结构如下：

.. code-block:: text

    ├── data
        ├── 20190720              # 训练数据的日期
            ├── 0000              # 训练数据的时间（第 1 个分片），0 时 0 分 - 0 时 5 分时间内的数据
                ├── data_part1    # 具体的训练数据文件
                ├── ......
            ├── 0005              # 训练数据的时间（第 2 个分片），0 时 5 分 - 0 时 10 分时间内的数据
                ├── data_part1    # 具体的训练数据文件
                ├── ......
            ├── 0010              # 训练数据的时间（第 3 个分片），0 时 10 分 - 0 时 15 分时间内的数据
                ├── data_part1    # 具体的训练数据文件
                ├── ......
            ├── ......
            ├── 2355              # 训练数据的时间（该日期下最后 1 个分片），23 时 55 分 - 24 时时间内的数据
                ├── data_part1    # 具体的训练数据文件
                ├── ......

根据 split_interval 和 split_per_pass 这两个配置项，在训练之前生成每个 Pass 所需要的数据分片列表，具体实现如下：

.. code-block:: python

    # 该方法定义在 tools/utils/static_ps/flow_helper.py 中
    def get_online_pass_interval(split_interval, split_per_pass,
                                is_data_hourly_placed):
        split_interval = int(split_interval)
        split_per_pass = int(split_per_pass)
        splits_per_day = 24 * 60 // split_interval
        pass_per_day = splits_per_day // split_per_pass
        left_train_hour = 0
        right_train_hour = 23

        start = 0
        split_path = []
        for i in range(splits_per_day):
            h = start // 60
            m = start % 60
            if h < left_train_hour or h > right_train_hour:
                start += split_interval
                continue
            if is_data_hourly_placed:
                split_path.append("%02d" % h)
            else:
                split_path.append("%02d%02d" % (h, m))
            start += split_interval

        start = 0
        online_pass_interval = []
        for i in range(pass_per_day):
            online_pass_interval.append([])
            for j in range(start, start + split_per_pass):
                online_pass_interval[i].append(split_path[j])
            start += split_per_pass

        return online_pass_interval

    # 根据 split_interval 和 split_per_pass，在训练之前生成每个 Pass 所需要的数据分片列表
    self.online_intervals = get_online_pass_interval(
              self.split_interval, self.split_per_pass, False)

例如：split_interval 配置为 5，split_per_pass 配置为 2，即数据分片时间间隔为 5 分钟，每个 Pass 的训练数据包含 2 个分片，则 online_intervals 数组的具体值为：[[0000, 0005], [0005, 0010], ..., [2350, 2355]]。

2.2 数据等待方式
""""""""""""

如果在训练过程中，需要等待数据准备完成，则需要配置 data_donefile 选项。

开启数据等待后，当数据目录中存在 data_donefile 配置对应的文件（一般是一个空文件）时，才会对该目录下的数据执行后续操作，否则，等待 data_sleep_second 时间后，重新探测是否存在 data_donefile 文件。

2.3 数据拆分方式
""""""""""""

由于参数服务器中存在多个训练 Worker，为保证每个训练 Worker 只训练数据集中的一部分，需要使用 ``fleet.util.get_file_shard()`` 对训练集进行拆分

.. code-block:: python

    # 该方法定义在 tools/utils/static_ps/flow_helper.py 中
    def file_ls(path_array, client):
        # 获取 path 数组下的所有文件
        # 如果数据存在 hdfs/afs 上，需要使用 hadoop_client
        result = []
        for path in path_array:
            if is_local(path):
                cur_path = os.listdir(path)
            else:
                cur_path = client.ls_dir(path)[1]
            if len(cur_path) > 0:
                result += [os.path.join(path, i) for i in cur_path]
        logger.info("file ls result = {}".format(result))
        return result

    cur_path = []
    for i in self.online_intervals[pass_index - 1]:
        # p 为一个具体的数据分片目录，例如："data/20190720/0000"
        p = os.path.join(train_data_path, day, str(i))
        if self.data_donefile:
          # 数据等待策略生效，如果目录下无 data_donefile 文件，需等待 data_sleep_second 后再探测
          cur_donefile = os.path.join(p, self.data_donefile)
          data_ready(cur_donefile, self.data_sleep_second,
                    self.hadoop_client)
        # cur_path 存储当前 Pass 下的所有数据目录，对应一个或多个数据分片文件夹
        # 例如：["data/20190720/0000", "data/20190720/0005"]
        cur_path.append(p)

    # 获取当前数据分片下的所有数据文件
    global_file_list = file_ls(cur_path, self.hadoop_client)
    # 将数据文件拆分到每个 Worker 上
    my_file_list = fleet.util.get_file_shard(global_file_list)

2.4 数据读取
""""""""""""

流式训练通常采用 InMemoryDataset 来读取数据，InMemoryDataset 会将当前 Worker 中的所有数据全部加载到内存，并支持秒级全局打散等功能。

.. code-block:: python

    # 创建 InMemoryDataset
    dataset = paddle.distributed.InMemoryDataset()

    # InMemoryDataset 初始化
    dataset.init(use_var=self.input_data,
                 pipe_command=self.pipe_command,
                 batch_size=batch_size,
                 thread_num=thread_num)

    # 设置文件列表为拆分到当前 Worker 的 file_list
    dataset.set_filelist(my_file_list)

    # 将训练数据加载到内存
    dataset.load_into_memory()
    # 数据全局打散
    dataset.global_shuffle(fleet, shuffle_thread_num)
    # 获取当前 Worker 在全局打散之后的训练数据样例数
    shuffle_data_size = dataset.get_shuffle_data_size(fleet)

    # 省略具体的训练过程

    # 在当前 Pass 训练结束后，InMemoryDataset 需调用 release_memory()方法释放内存
    dataset.release_memory()

2.5 数据预读取
""""""""""""

由于数据读取是 IO 密集型任务，而模型训练是计算密集型任务，为进一步提升整体训练性能，可以将数据读取和模型训练两个阶段做 overlap 处理，即在上一个 pass 训练过程中预读取下一个 pass 的数据。

具体地，可以使用 dataset 的以下两个 API 进行数据预读取操作：
1. ``preload_into_memory()`` ：创建 dataset 后，使用该 API 替换 ``load_into_memory()`` ，在当前 pass 的训练过程中，预读取下一个 pass 的训练数据。
2. ``wait_preload_done()`` ：在下一个 pass 训练之前，调用 ``wait_preload_done()`` ，等待 pass 训练数据全部读取完毕，进行训练。

3 模型训练及预测
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

模型训练及预测使用 ``exe.train_from_dataset()`` 和 ``exe.infer_from_dataset()`` 接口即可，本章节讲解一下在训练和预测过程中计算分布式指标上的一些细节以及如何利用 debug 模式下的 dump 功能打印模型计算的中间结果。

3.1 分布式指标计算
""""""""""""

在之前的参数服务器概述中曾经提到，由于参数服务器存在多个训练节点，因此在计算指标时，需要汇总所有节点的全量数据，进行全局指标计算。

除此之外，分布式全局指标计算还需要注意以下两点：

1. 参数服务器的训练节点一般会存在多个线程同时进行训练，而所有线程共享指标计算所需的中间变量，这就可能导致中间变量的累计计数不准确，因此需要让每个线程拥有自己独立的中间变量。
2. 指标计算所需的中间变量在整个训练过程中会持续累计计数，因此需要在合适的位置进行清零操作，避免当前指标计算受之前累计计数的影响。

同样是以 AUC 指标为例，全局 AUC 指标计算示例如下：

.. code-block:: python

    # 该方法定义在 tools/utils/static_ps/metric_helper.py 中
    def set_zero(var_name,
                 scope=fluid.global_scope(),
                 place=fluid.CPUPlace(),
                 param_type="int64"):
        # 对变量进行清零操作
        param = scope.var(var_name).get_tensor()
        param_array = np.zeros(param._get_dims()).astype(param_type)
        param.set(param_array, place)

    # 组网阶段，AUC 算子在计算 auc 指标同时，返回正负样例中间统计结果（stat_pos, stat_neg）
    auc, batch_auc, [batch_stat_pos, batch_stat_neg, stat_pos, stat_neg] = \
        paddle.static.auc(input=pred, label=label)

    strategy = fleet.DistributedStrategy()
    strategy.a_sync = True

    # 获取计算指标所需的中间变量的 name 列表，并将其配置到 strategy 的 stat_var_names 选项中
    stat_var_names = [stat_pos.name, stat_neg.name]
    strategy.trainer_desc_configs = {"stat_var_names": stat_var_names}

    # 省略具体训练过程

    # 训练结束后，利用 AUC 算子返回的中间计算结果，以及 fleet 提供的分布式指标计算接口，完成全局 AUC 计算。
    global_auc = fleet.metrics.auc(stat_pos, stat_neg)

    # 指标计算所需的中间变量清零
    set_zero(stat_pos.name)
    set_zero(stat_neg.name)

3.2 Dump 功能
""""""""""""

Debug 模式下的 dump 功能主要为了解决以下两个问题：

1. 在训练过程中希望打印模型计算的中间结果，用于监控模型是否收敛等情况。
2. 为减轻线上推理服务的计算压力，在召回或者匹配模型中，一般需要将 doc 侧的向量预先计算出来，灌入向量搜索引擎（例如 milvus）中。因此需要在流式训练过程中加入预测阶段打印 doc 侧的向量计算结果。

.. code-block:: python

    # 该方法定义在 tools/utils/static_ps/program_helper.py 中
    def set_dump_config(program, dump_config):
        # 配置 dump 相关信息
        if dump_config.get("dump_fields_path") is not None:
            # 打印出的中间结果存放路径
            program._fleet_opt["dump_fields_path"] = dump_config.get(
                "dump_fields_path")
        if dump_config.get("dump_fields") is not None:
            # 需要打印的中间层变量名
            program._fleet_opt["dump_fields"] = dump_config.get("dump_fields")
        if dump_config.get("dump_param") is not None:
            # 需要打印的参数名
            program._fleet_opt["dump_param"] = dump_config.get("dump_param")

    # dataset 需要设置 parse_ins_id 和 parse_content 为 True
    # 同时，输入数据也需要在最前面增加 ins_id 和 content 两个字段，用来标识具体的样例
    dataset.set_parse_ins_id(True)
    dataset.set_parse_content(True)

    # 在训练或者预测前配置 dump 信息
    dump_fields_dir = "dump_data"
    # dump 出的中间结果存放路径
    dump_fields_path = "{}/{}/{}".format(dump_fields_dir, day, pass_index)
    # 需要 dump 的中间变量，具体定义参考 static_model.py 和 net.py
    dump_fields = [var.name for var in self.infer_dump_fields]
    # 调用 set_dump_config 配置 dump 信息
    set_dump_config(paddle.static.default_main_program(), {
        "dump_fields_path": dump_fields_path,
        "dump_fields": dump_fields
    })

    # 预测
    self.exe.infer_from_dataset(
        program=paddle.static.default_main_program(),
        dataset=cur_dataset,
        fetch_list=fetch_vars,
        fetch_info=fetch_info,
        print_period=print_step,
        debug=debug)


4 模型保存
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

为实现流式训练中的增量训练及线上推理部署，在训练过程中，需要保存几种不同类型的模型。

4.1 明文模型
""""""""""""

明文模型（checkpoint model）主要用于增量训练中的模型加载。在流式训练中，由于数据、资源等问题，一直在运行的训练程序可能会挂掉，这时候需要加载之前已经保存好的明文模型，再此基础上继续进行后续的增量训练。

明文模型的保存，由 0 号节点发送保存请求给所有服务节点，服务节点以明文形式保存模型全量的稀疏参数和稠密参数以及优化器状态。

另外，还有一种特殊的明文模型，叫作 batch_model，通常在每天数据训练结束后保存，与明文模型最大的区别在于，保存 batch_model 之前一般需要调用 ``fleet.shrink()`` 方法，删除掉一些长久不出现或者出现频率极低的稀疏特征。

.. code-block:: python

    # 该方法定义在 tools/utils/static_ps/flow_helper.py 中
    def save_model(exe, output_path, day, pass_id, mode=0):
        # 保存明文模型，具体目录为 output_path/day/pass_id，例如：output_path/20190720/6
        day = str(day)
        pass_id = str(pass_id)
        suffix_name = "/%s/%s/" % (day, pass_id)
        model_path = output_path + suffix_name
        fleet.save_persistables(exe, model_path, None, mode=mode)

    # 该方法定义在 tools/utils/static_ps/flow_helper.py 中
    def save_batch_model(exe, output_path, day):
        # 保存 batch_model，具体目录为 output_path/day/0，例如：output_path/20190721/0
        day = str(day)
        suffix_name = "/%s/0/" % day
        model_path = output_path + suffix_name
        fleet.save_persistables(exe, model_path, mode=3)

    for pass_id in range(1, 1 + len(self.online_intervals)):
        # 分 Pass 训练，省略具体训练过程

        if pass_id % self.checkpoint_per_pass == 0:
            # 在到达配置的 Pass 时，调用 save_model 保存明文模型
            save_model(self.exe, self.save_model_path, day, pass_id)

    # 一天数据训练完成
    # 调用 shrink 删除某些稀疏参数
    fleet.shrink()

    next_day = get_next_day(day)
    # 调用 save_batch_model 保存 batch_model
    save_batch_model(self.exe, self.save_model_path, next_day)

4.2 推理模型
""""""""""""

推理模型（inference model）主要用于线上推理部署。整个推理模型由以下三个部分组成：

1. 推理网络：由训练网络裁剪而来，一般来说，推理网络输入为 embedding 层的输出，网络输出为 label 的预估值，即推理网络中不包括 embedding 层，也不包括损失值和指标计算。
2. 稠密参数：稠密参数由某个训练节点（一般是 0 号训练节点）以二进制方式保存在该节点的本地磁盘。
3. 稀疏参数：由于搜索推荐场景下的稀疏参数通常量级巨大，因此一般配送到专用的 KV 存储中（例如 cube、redis）。稀疏参数的保存由 0 号节点发送请求给所有服务节点，服务节点可将稀疏参数通过具体的 converter 保存成线上 KV 存储所需的格式。同时为节省线上推理所需的存储空间，保存的稀疏参数可能并非全量，有一定的过滤逻辑。

稀疏参数进一步区分为 base 模型和 delta 模型。base 模型通常一天保存一次，在 base 模型的基础上，在一天之内，每间隔一段时间保存一个 delta 模型。

.. code-block:: python

    # 该方法定义在 tools/utils/static_ps/flow_helper.py 中
    def save_inference_model(output_path, day, pass_id, exe, feed_vars, target_vars, client):
        if pass_id != -1:
            # mode=1，保存 delta 模型
            mode = 1
            suffix_name = "/%s/delta-%s/" % (day, pass_id)
            model_path = output_path.rstrip("/") + suffix_name
        else:
            # mode=2，保存 base 模型
            mode = 2
            suffix_name = "/%s/base/" % day
            model_path = output_path.rstrip("/") + suffix_name
        fleet.save_inference_model(
            exe,
            model_path, [feed.name for feed in feed_vars],
            target_vars,
            mode=mode)
        if not is_local(model_path) and fleet.is_first_worker():
            client.upload_dir("./dnn_plugin", model_path)
        fleet.barrier_worker()

    # 定义推理裁剪网络的输入和输出，具体定义参考 static_model.py 和 net.py
    self.inference_feed_vars = model.inference_feed_vars
    self.inference_target_var = model.inference_target_var
    for pass_id in range(1, 1 + len(self.online_intervals)):
        # 分 Pass 训练，省略具体训练过程

        if pass_id % self.save_delta_frequency == 0:
            # 在到达配置的 Pass 时，调用 save_xbox_model 保存 delta 推理模型
            save_inference_model(self.save_model_path, day, pass_id,
                                 self.exe, self.inference_feed_vars,
                                 self.inference_target_var,
                                 self.hadoop_client)

    # 一天数据训练完成
    # 调用 shrink 删除某些稀疏参数
    fleet.shrink()

    next_day = get_next_day(day)
    # 由 0 号节点调用 save_xbox_model 保存 base 推理模型
    save_inference_model(self.save_model_path, next_day, -1,
                         self.exe, self.inference_feed_vars,
                         self.inference_target_var,
                         self.hadoop_client)

5 稀疏参数高级功能
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

为进一步提升模型效果，降低存储空间，关于稀疏参数提供了一系列高级功能，下面逐一进行介绍相关的功能和配置。

具体配置详情可参考\ `slot_dnn 中的 config_online 配置文件 <https://github.com/PaddlePaddle/PaddleRec/blob/master/models/rank/slot_dnn/config_online.yaml>`_\中的 table_parameters 部分，如果用户不配置相关选项，框架将使用默认值。

为使用高级功能，需要配置稀疏参数相应的 table 及 accessor：

.. csv-table::
    :header: "名称", "类型", "取值", "是否必须", "作用描述"
    :widths: 10, 5, 5, 5, 30

    "table_class", "string", "MemorySparseTable", "是", "存储 embedding 的 table 名称"
    "accessor_class", "string", "SparseAccessor", "是", "获取 embedding 的 accessor 名称"

5.1 特征频次计算
""""""""""""

server 端会根据特征的 show 和 click 计算一个频次得分，用于判断该特征 embedding 是否可以扩展、保存等，具体涉及到的配置如下：

.. csv-table::
    :header: "名称", "类型", "取值", "默认值", "是否必须", "作用描述"
    :widths: 10, 5, 5, 5, 5, 30

    "nonclk_coeff", "float", "任意", "0.1", "是", "特征展现但未点击对应系数"
    "click_coeff", "float", "任意", "1.0", "是", "特征点击对应系数"

具体频次 score 计算公式如下：
score = click_coeff * click + noclick_coeff * (click - show)

5.2 特征 embedding 准入
""""""""""""

特征 embedding 初始情况下，只会生成一维 embedding，其余维度均为 0，当特征的频次 score 大于等于扩展阈值时，才会扩展出剩余维度，具体涉及到的配置如下：

.. csv-table::
    :header: "名称", "类型", "取值", "默认值", "是否必须", "作用描述"
    :widths: 10, 5, 5, 5, 5, 30

    "embedx_threshold", "int", "任意", "0", "是", "特征 embedding 扩展阈值"
    "embedx_dim", "int", "任意", "组网 sparse_embedding 层参数 size 第二维值-1", "是", "特征 embedding 扩展维度"
    "fea_dim", "int", "任意", "组网 sparse_embedding 层参数 size 第二维值+2", "是", "特征 embedding 总维度"

需要注意的是：

1. 特征 embedding 的实际维度（组网 sparse_embedding 层参数 size 第二维值）为 1 + embedx_dim，即一维初始 embedding + 扩展 embedding。
2. 特征总维度包括 show 和 click，因此 fea_dim = embedx_dim + 3。

5.3 特征 embedding 淘汰
""""""""""""

为避免稀疏特征无限增加，一般每天的数据训练完成后，会调用 ``fleet.shrink()`` 方法，删除掉一些长久不出现或者出现频率极低的稀疏特征，具体涉及到的配置如下：

.. csv-table::
    :header: "名称", "类型", "取值", "默认值", "是否必须", "作用描述"
    :widths: 10, 5, 5, 5, 5, 30

    "show_click_decay_rate", "float", "[0,1]", "1", "是", "调用 shrink 函数时，show 和 click 会根据该配置进行衰减"
    "delete_threshold", "float", "任意", "0", "是", "特征频次 score 小于该阈值时，删除该特征"
    "delete_after_unseen_days", "int", ">0", "30", "是", "特征未出现天数大于该阈值时，删除该特征"

5.4 特征 embedding 保存
""""""""""""

为降低模型保存的磁盘占用及耗时，在保存 base/delta 模型时，可以去掉部分出现频率不高的特征，具体涉及到的配置如下：

.. csv-table::
    :header: "名称", "类型", "取值", "默认值", "是否必须", "作用描述"
    :widths: 10, 5, 5, 5, 5, 30

    "base_threshold", "float", "任意", "0", "是", "特征频次 score 大于等于该阈值才会在 base 模型中保存"
    "delta_threshold", "float", "任意", "0", "是", "从上一个 delta 模型到当前 delta 模型，特征频次 score 大于等于该阈值才会在 delta 模型中保存"
    "delta_keep_days", "int", "任意", "16", "是", "特征未出现天数小于等于该阈值才会在 delta 模型中保存"
    "converter", "string", "任意", "", "否", "base/delta 模型转换器（对接线上推理 KV 存储）"
    "deconverter", "string", "任意", "", "否", "base/delta 模型解压器"

5.5 参数优化算法
""""""""""""

稀疏参数(sparse_embedding)优化算法配置，分为一维 embedding 的优化算法(embed_sgd_param)和扩展 embedding 的优化算法(embedx_sgd_param)：

.. csv-table::
    :header: "名称", "类型", "取值", "默认值", "是否必须", "作用描述"
    :widths: 10, 5, 5, 5, 5, 30

    "name", "string", "SparseAdaGradSGDRule", "SparseAdaGradSGDRule", "是", "优化算法名称"
    "learning_rate", "float", "任意", "0.05", "是", "学习率"
    "initial_g2sum", "float", "任意", "3.0", "是", "g2sum 初始值"
    "initial_range", "float", "任意", "0.0001", "是", "embedding 初始化范围[-initial_range,initial_range]"
    "weight_bounds", "list(float)", "任意", "[-10.0,10.0]", "是", "embedding 在训练过程中的范围"

稠密参数优化算法配置：

.. csv-table::
    :header: "名称", "类型", "取值", "默认值", "是否必须", "作用描述"
    :widths: 10, 5, 5, 5, 5, 30

    "adam_d2sum", "bool", "任意", "否", "是", "是否使用新的稠密参数优化算法"
