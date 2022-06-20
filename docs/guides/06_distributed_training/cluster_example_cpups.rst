
..  _cluster_example_cpups:

CPUPS流式训练示例
-------------------------

在之前的参数服务器概述中曾经提到，由于推荐搜索场景的特殊性，在训练过程中使用到的训练数据通常不是固定的集合，而是随时间流式的加入到训练过程中，这种训练方式称为流式训练。

与传统的固定数据集的训练相比，在大规模稀疏参数场景下，流式训练有如下特点：

1. 在线上系统提供服务的过程中，训练程序一直运行，等待线上服务产生的数据，并在数据准备完成后进行增量训练。
2. 数据一般按时间组织，每个Pass的训练数据对应一个或几个时间分片（文件夹），并在该时间分片的数据生成结束后，在对应时间分片的文件夹中添加一个空文件用于表示数据准备完成。
3. 线上服务产生的数据不断进入训练系统，导致稀疏参数不断增加，在模型精度不受影响的情况下，为控制模型总存储，增加稀疏参数统计量自动统计、稀疏参数准入、退场、增量保存等功能。
4. 

在学习流式训练具体使用方法之前，建议先详细阅读参数服务器快速开始章节，了解参数服务器的基本使用方法。

流式训练的完整代码示例参见：\ `PaddleRec流式训练 <https://github.com/PaddlePaddle/PaddleRec/blob/master/tools/static_ps_online_trainer.py>`_\，其中所使用到的模型及配置参见：\ `slot_dnn <https://github.com/PaddlePaddle/PaddleRec/tree/master/models/rank/slot_dnn>`_\，重点关注其中的流式训练配置文件config_online.yaml，及模型组网net.py和static_model.py。

1 模型组网

为实现稀疏参数统计量自动统计，在组网时需要注意以下两点：
1. embedding层需使用paddle.static.nn.sparse_embedding，该算子为大规模稀疏参数模型特定的embedding算子，支持对稀疏参数统计量自动统计。
2. 稀疏参数的统计量，目前特指稀疏特征的展现量(show)和点击量(click)，在组网时定义两个变量，指明特征是否展现和点击，通常取值为0或者1，sparse_embedding中通过entry参数传入一个ShowClickEntry，指明这两个变量(show和click)的名字。

.. code-block:: python

  # net.py
  # 构造ShowClickEntry，指明展现和点击对应的变量名
  self.entry = paddle.distributed.ShowClickEntry("show", "click")
  emb = paddle.static.nn.sparse_embedding(
      input=s_input,
      size=[self.dict_dim, self.emb_dim],
      padding_idx=0,
      entry=self.entry,   # 在sparse_embedding中传入entry
      param_attr=paddle.ParamAttr(name="embedding"))

  # static_model.py
  # 构造show/click对应的data，变量名需要与entry中的名称一致
  show = paddle.static.data(
      name="show", shape=[None, 1], dtype="int64")
  label = paddle.static.data(
      name="click", shape=[None, 1], dtype="int64")

2 数据读取
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

本章节主要讲解流式训练中的数据组织形式、数据等待方式、数据拆分方式等内容，涉及到的概念如下：

1. 数据分片：将数据按照固定的时间间隔组织成多个分片，一段时间间隔中的所有训练数据称为一个数据分片。
2. Pass：把当前训练集的所有数据全部跑一遍，通常一个Pass对应一个或多个数据分片。

涉及到的具体配置如下：

|             名称              |     类型     |     取值    | 是否必须 |                               作用描述                    |
| :---------------------------: | :----------: | :--------: | :------: | :-----------------------------------------------------: |
|        train_data_dir         |  string  |     任意       |    是    |       训练数据所在目录（如果数据存于afs或hdfs上，以afs:/或hdfs:/开头）      |
|        split_interval         |  int     |     任意       |    是    |       数据落盘分片间隔时间（分钟）                          |
|        split_per_pass         |  int     |     任意       |    是    |       训练一个Pass包含多少个分片的数据                      |
|        start_day              |  string  |     任意       |    是    |       训练开始的日期（例：20190720）                       |
|        end_day                |  string  |     任意       |    是    |       训练结束的日期（例：20190720）                       |
|        data_donefile          |  string  |     任意       |    否    |       用于探测当前分片数据是否落盘完成的标识文件             |
|        data_sleep_second      |  int     |     任意       |    否    |       当前分片数据尚未完成的等待时间                        |

2.1 数据组织形式
""""""""""""

在训练数据目录下，再建立两层目录，第一层目录对应训练数据的日期（8位），第二层目录对应训练数据的具体时间（4位，前两位为小时，后两位为分钟），并且需要与配置文件中的split_interval配置对应。
例如：train_data_dir配置为“data”目录，split_interval配置为5，则具体的目录结构如下：

```txt
├── data
    ├── 20190720              # 训练数据的日期
        ├── 0000              # 训练数据的时间（第1个分片），0时0分-0时5分时间内的数据
            ├── data_part1    # 具体的训练数据文件
            ├── ......    
        ├── 0005              # 训练数据的时间（第2个分片），0时5分-0时10分时间内的数据
            ├── data_part1    # 具体的训练数据文件
            ├── ......
        ├── 0010              # 训练数据的时间（第3个分片），0时10分-0时15分时间内的数据
            ├── data_part1    # 具体的训练数据文件
            ├── ......
        ├── ......
        ├── 2355              # 训练数据的时间（该日期下最后1个分片），23时55分-24时时间内的数据
            ├── data_part1    # 具体的训练数据文件
            ├── ......
```

根据split_interval和split_per_pass这两个配置项，在训练之前生成每个Pass所需要的数据分片列表，具体实现如下：

.. code-block:: python

  # 该方法定义在tools/utils/static_ps/flow_helper.py中
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

  # 根据split_interval和split_per_pass，在训练之前生成每个Pass所需要的数据分片列表
  self.online_intervals = get_online_pass_interval(
            self.split_interval, self.split_per_pass, False)

例如：split_interval配置为5，split_per_pass配置为2，即数据分片时间间隔为5分钟，每个Pass的训练数据包含2个分片，则online_intervals数组的具体值为：[[0000, 0005], [0005, 0010], ..., [2350, 2355]]。

2.2 数据等待方式
""""""""""""

如果在训练过程中，需要等待数据准备完成，则需要配置data_donefile选项。

开启数据等待后，当数据目录中存在data_donefile配置对应的文件（一般是一个空文件）时，才会对该目录下的数据执行后续操作，否则，等待data_sleep_second时间后，重新探测是否存在data_donefile文件。

2.3 数据拆分方式
""""""""""""

由于参数服务器中存在多个训练Worker，为保证每个训练Worker只训练数据集中的一部分，需要使用 ``fleet.util.get_file_shard()`` 对训练集进行拆分

.. code-block:: python

  # 该方法定义在tools/utils/static_ps/flow_helper.py中
  def file_ls(path_array, client):
    # 获取path数组下的所有文件
    # 如果数据存在hdfs/afs上，需要使用hadoop_client
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
    # p为一个具体的数据分片目录，例如："data/20190720/0000"
    p = os.path.join(train_data_path, day, str(i))
    if self.data_donefile:
      # 数据等待策略生效，如果目录下无data_donefile文件，需等待data_sleep_second后再探测
      cur_donefile = os.path.join(p, self.data_donefile)
      data_ready(cur_donefile, self.data_sleep_second,
                self.hadoop_client)
    # cur_path存储当前Pass下的所有数据目录，对应一个或多个数据分片文件夹
    # 例如：["data/20190720/0000", "data/20190720/0005"]
    cur_path.append(p)
    
  # 获取当前数据分片下的所有数据文件
  global_file_list = file_ls(cur_path, self.hadoop_client)
  # 将数据文件拆分到每个Worker上
  my_file_list = fleet.util.get_file_shard(global_file_list)

2.4 数据读取
""""""""""""

流式训练通常采用InMemoryDataset来读取数据，InMemoryDataset会将当前Worker中的所有数据全部加载到内存，并支持秒级全局打散等功能。

.. code-block:: python

  # 创建InMemoryDataset
  dataset = paddle.distributed.InMemoryDataset()
  
  # InMemoryDataset初始化
  dataset.init(use_var=self.input_data, 
                pipe_command=self.pipe_command, 
                batch_size=batch_size, 
                thread_num=thread_num)
  
  # 设置文件列表为拆分到当前Worker的file_list
  dataset.set_filelist(my_file_list)
  
  # 将训练数据加载到内存
  dataset.load_into_memory()
  # 数据全局打散
  dataset.global_shuffle(fleet, shuffle_thread_num)
  # 获取当前Worker在全局打散之后的训练数据样例数
  shuffle_data_size = dataset.get_shuffle_data_size(fleet)

  # 省略具体的训练过程

  # 在当前Pass训练结束后，InMemoryDataset需调用release_memory()方法释放内存
  dataset.release_memory()
  

3 模型训练及预测
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

4 模型保存
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

5 稀疏参数高级功能
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

为进一步提升模型效果，降低存储空间，关于稀疏参数提供了一系列高级功能，下面逐一进行介绍相关的功能和配置。

具体配置详情可参考\ `slot_dnn中的config_online配置文件 <https://github.com/PaddlePaddle/PaddleRec/blob/master/models/rank/slot_dnn/config_online.yaml>`_\中的table_parameters部分，如果用户不配置相关选项，框架将使用默认值。

为使用高级功能，需要配置稀疏参数相应的table及accessor：

|             名称              |     类型     |                           取值                            | 是否必须 |                               作用描述                               |
| :---------------------------: | :----------: | :-------------------------------------------------------: | :------: | :------------------------------------------------------------------: |
|         table_class           |    string    |     MemorySparseTable           |    是    |        存储embedding的table名称     |
|         accessor_class        |    string    |     SparseAccessor              |    是    |       获取embedding的accessor名称       |

5.1 特征频次计算
""""""""""""

server端会根据特征的show和click计算一个频次得分，用于判断该特征embedding是否可以扩展、保存等，具体涉及到的配置如下：

|             名称              |     类型     |         取值         |      默认值   |是否必须 |                               作用描述                               |
| :---------------------------: | :----------: | :--------------------| :---------: | :------: | :------------------------------------------------------------------: |
|         nonclk_coeff         |    float    |           任意         |    0.1        |    是    |                            特征展现但未点击对应系数                            |
|         click_coeff          |    float    |           任意         |    1.0       |    是    |                            特征点击对应系数                            |

具体频次score计算公式如下：  
score = click_coeff * click + noclick_coeff * (click - show)

5.2 特征embedding准入
""""""""""""

特征embedding初始情况下，只会生成一维embedding，其余维度均为0，当特征的频次score大于等于扩展阈值时，才会扩展出剩余维度，具体涉及到的配置如下：
|             名称              |     类型      |      取值         |      默认值                                      | 是否必须 |            作用描述                |
| :---------------------------: | :----------: | :---------------: | :-----------------------------------------------: | :------: | :--------------------------: |
|         embedx_threshold      |    int       |       任意         |    0                                            |    是    |    特征embedding扩展阈值       |
|         embedx_dim            |    int       |       任意         |    组网sparse_embedding层参数size第二维值 - 1    |    是    |     特征embedding扩展维度         |
|         fea_dim               |    int       |       任意         |    组网sparse_embedding层参数size第二维值 + 2    |    是    |     特征embedding总维度          |

需要注意的是：

1. 特征embedding的实际维度为1 + embedx_dim，即一维初始embedding + 扩展embedding。
2. 特征总维度包括show和click，因此fea_dim = embedx_dim + 3。

5.3 特征embedding淘汰
""""""""""""

为避免稀疏特征无限增加，一般每天的数据训练完成后，会调用shrink函数删除掉一些长久不出现或者出现频率极低的特征，具体涉及到的配置如下：

|             名称              |     类型     |       取值            |      默认值   | 是否必须 |                               作用描述                               |
| :---------------------------: | :----------: | :------------------: | :---------: | :------: | :------------------------------------------------------------------: |
|    show_click_decay_rate      |    float    |       [0, 1]           |    1    |    是    |   调用shrink函数时，show和click会根据该配置进行衰减               |
|    delete_threshold           |    float    |       任意             |    0    |    是    |       特征频次score小于该阈值时，删除该特征                 |
|    delete_after_unseen_days   |    int      |        >0             |    30    |    是    |       特征未出现天数大于该阈值时，删除该特征                 |


5.4 特征embedding保存
""""""""""""

为降低模型保存的磁盘占用及耗时，在保存base/delta模型时，可以去掉部分出现频率不高的特征，具体涉及到的配置如下：
|             名称              |     类型     |       取值         |      默认值   | 是否必须 |                               作用描述                               |
| :---------------------------: | :----------: | :----------------: | :---------: | :------: | :------------------------------------------------------------------: |
|        base_threshold         |    float    |      任意            |    0    |    是    |       特征频次score大于等于该阈值才会在base模型中保存                            |
|        delta_threshold        |    float    |      任意            |    0    |    是    |   从上一个delta模型到当前delta模型，<br>特征频次score大于等于该阈值才会在delta模型中保存        |
|        delta_keep_days        |    int      |      任意            |    16    |    是    |   特征未出现天数小于等于该阈值才会在delta模型中保存               |
|        converter              |    string    |     任意            |    ""    |    否   |   base/delta模型转换器（对接线上推理KV存储）            |
|        deconverter            |    string    |     任意            |    ""    |    否    |   base/delta模型解压器               |


5.5 参数优化算法
""""""""""""

稀疏参数(sparse_embedding)优化算法配置，分为一维embedding的优化算法(embed_sgd_param)和扩展embedding的优化算法(embedx_sgd_param)：
|             名称              |     类型     |                           取值                            |      默认值   | 是否必须 |                               作用描述                               |
| :---------------------------: | :----------: | :-------------------------------------------------------: | :------: | :------: | :------------------------------------------------------------------: |
|             name              |    string    |    SparseAdaGradSGDRule<br>SparseNaiveSGDRule<br>SparseAdamSGDRule<br>StdAdaGradSGDRule      |    SparseAdaGradSGDRule    |    是    |       优化算法名称                 |
|       learning_rate           |    float    |    任意                  |    0.05   |    是    |       学习率                 |
|       initial_g2sum           |    float    |    任意                  |    3.0    |    是    |       g2sum初始值                 |
|       initial_range           |    float    |    任意                  |    0.0001    |    是    |       embedding初始化范围[-initial_range, initial_range]          |
|       weight_bounds           |    list(float)    |    任意                  |    [-10.0, 10.0]    |    是    |    embedding在训练过程中的范围        |

稠密参数优化算法配置：
|             名称              |     类型     |                           取值                            | 是否必须 |                               作用描述                               |
| :---------------------------: | :----------: | :-------------------------------------------------------: | :------: | :------------------------------------------------------------------: |
|             adam_d2sum              |    bool    |    任意                        |    是    |       是否使用新的稠密参数优化算法                 |

DistributedStrategy：三个模式介绍
distributed_optimizer：切图大体逻辑，增加pull和push算子

数据处理：
fleet.util.get_file_shard：数据拆分
重点介绍InmemoryDataset：load_into_memory, release_memory, global_shuffle

训练/预测：dump

指标计算：是否解释stat_var_name

模型保存：inference model需要与线上推理结合，稀疏参数入cube，模型裁剪

