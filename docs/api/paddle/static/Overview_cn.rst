.. _cn_overview_static:

paddle.static
---------------------

paddle.static 下的 API 为飞桨静态图专用 API。具体如下：

-  :ref:`Program 相关 API <about_program>`
-  :ref:`Executor 相关 API <about_executor>`
-  :ref:`组网相关 API <about_nn>`
-  :ref:`io 相关 API <about_io>`
-  :ref:`变量相关 API <about_variable>`
-  :ref:`运行设备相关 API <about_device>`
-  :ref:`评估指标相关 API <about_metrics>`
-  :ref:`其他 API <about_others>`

.. _about_program:

Program 相关 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`append_backward <cn_api_paddle_static_append_backward>` ", "向 main_program 添加反向"
    " :ref:`default_main_program <cn_api_paddle_static_default_main_program>` ", "获取当前用于存储 OP 和 Tensor 描述信息的 `default main program` "
    " :ref:`default_startup_program <cn_api_paddle_static_default_startup_program>` ", "获取默认/全局的 `startup program` "
    " :ref:`Program <cn_api_paddle_static_Program>` ", "飞桨用 Program 动态描述整个计算图"
    " :ref:`program_guard <cn_api_paddle_static_program_guard>` ", "配合 with 语句将算子和变量添加进指定的 `main program` 和 `startup program` "
    " :ref:`set_program_state <cn_api_paddle_static_set_program_state>` ", "设置 Program 的参数和优化器信息"
    " :ref:`normalize_program <cn_api_paddle_static_normalize_program>` ", "根据指定的 feed_vars 和 fetch_vars，优化 program"

.. _about_executor:

Executor 相关 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`BuildStrategy <cn_api_paddle_static_BuildStrategy>` ", "控制 ParallelExecutor 中计算图的建造方法"
    " :ref:`CompiledProgram <cn_api_paddle_static_CompiledProgram>` ", "转化和优化 Program 或 Graph"
    " :ref:`Executor <cn_api_paddle_static_Executor>` ", "执行器"

.. _about_nn:

组网相关 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`batch_norm <cn_api_paddle_static_nn_batch_norm>` ", "Batch Normalization 方法"
    " :ref:`bilinear_tensor_product <cn_api_paddle_static_nn_bilinear_tensor_product>` ", "对两个输入执行双线性 Tensor 积"
    " :ref:`case <cn_api_paddle_static_nn_case>` ", "以 OP 的运行方式类似于 python 的 if-elif-elif-else"
    " :ref:`conv2d <cn_api_paddle_static_nn_conv2d>` ", "二维卷积层"
    " :ref:`conv2d_transpose <cn_api_paddle_static_nn_conv2d_transpose>` ", "二维转置卷积层"
    " :ref:`conv3d <cn_api_paddle_static_nn_conv3d>` ", "三维卷积层"
    " :ref:`conv3d_transpose <cn_api_paddle_static_nn_conv3d_transpose>` ", "三维转置卷积层"
    " :ref:`crf_decoding <cn_api_paddle_static_nn_crf_decoding>` ", "CRF Decode 层"
    " :ref:`data_norm <cn_api_paddle_static_nn_data_norm>` ", "数据正则化层"
    " :ref:`deform_conv2d <cn_api_paddle_static_nn_deform_conv2d>` ", "可变形卷积层"
    " :ref:`embedding <cn_api_paddle_static_nn_embedding>` ", "嵌入层"
    " :ref:`sparse_embedding <cn_api_paddle_static_nn_sparse_embedding>` ", "稀疏嵌入层"
    " :ref:`fc <cn_api_paddle_static_nn_fc>` ", "全连接层"
    " :ref:`group_norm <cn_api_paddle_static_nn_group_norm>` ", "Group Normalization 方法"
    " :ref:`instance_norm <cn_api_paddle_static_nn_instance_norm>` ", "Instance Normalization 方法"
    " :ref:`layer_norm <cn_api_paddle_static_nn_layer_norm>` ", "Layer Normalization 方法"
    " :ref:`multi_box_head <cn_api_paddle_static_nn_multi_box_head>` ", "SSD 检测头 "
    " :ref:`nce <cn_api_paddle_static_nn_nce>` ", "计算并返回噪音对比估计损失"
    " :ref:`prelu <cn_api_paddle_static_nn_prelu>` ", "prelu 激活函数"
    " :ref:`row_conv <cn_api_paddle_static_nn_row_conv>` ", "行卷积"
    " :ref:`spectral_norm <cn_api_paddle_static_nn_spectral_norm>` ", "Spectral Normalization 方法"
    " :ref:`switch_case <cn_api_paddle_static_nn_switch_case>` ", "类似于 c++的 switch/case"
    " :ref:`sequence_concat <cn_api_paddle_static_nn_sequence_concat>` ", "仅支持带有 LoD 信息的 Tensor ，通过 Tensor 的 LoD 信息将输入的多个 Tensor 进行连接，输出连接后的 Tensor"
    " :ref:`sequence_conv <cn_api_paddle_static_nn_sequence_conv>` ", "仅支持带有 LoD 信息的 Tensor，在给定的卷积参数下，对输入的变长序列 Tensor 进行卷积操作"
    " :ref:`sequence_enumerate <cn_api_paddle_static_nn_sequence_enumerate>` ", "仅支持带有 LoD 信息的 Tensor，枚举形状为 [d_1, 1] 的输入序列所有长度为 win_size 的子序列，生成一个形状为 [d_1, win_size] 的新序列，需要时以 pad_value 填充"
    " :ref:`sequence_expand <cn_api_paddle_static_nn_sequence_expand>` ", "仅支持带有 LoD 信息的 Tensor，根据输入 y 的第 ref_level 层 lod 对输入 x 进行扩展"
    " :ref:`sequence_expand_as <cn_api_paddle_static_nn_sequence_expand_as>` ", "仅支持带有 LoD 信息的 Tensor，根据输入 y 的第 0 级 lod 对输入 x 进行扩展"
    " :ref:`sequence_first_step <cn_api_paddle_static_nn_sequence_first_step>` ", "仅支持带有 LoD 信息的 Tensor，对输入的 Tensor，在最后一层 lod_level 上，选取其每个序列的第一个时间步的特征向量作为池化后的输出向量"
    " :ref:`sequence_last_step <cn_api_paddle_static_nn_sequence_last_step>` ", "仅支持带有 LoD 信息的 Tensor，对输入的 Tensor，在最后一层 lod_level 上，选取其每个序列的最后一个时间步的特征向量作为池化后的输出向量"
    " :ref:`sequence_pad <cn_api_paddle_static_nn_sequence_pad>` ", "仅支持带有 LoD 信息的 Tensor，将同一 batch 中的序列填充到一个一致的长度（由 maxlen 指定）"
    " :ref:`sequence_pool <cn_api_paddle_static_nn_sequence_pool>` ", "仅支持带有 LoD 信息的 Tensor，对输入的 Tensor 进行指定方式的池化操作"
    " :ref:`sequence_reshape <cn_api_paddle_static_nn_sequence_reshape>` ", "仅支持带有 LoD 信息的 Tensor，对输入的 Tensor 进行指定方式的变形操作"
    " :ref:`sequence_reverse <cn_api_paddle_static_nn_sequence_reverse>` ", "仅支持带有 LoD 信息的 Tensor，对输入的 Tensor，在每个序列上进行反转"
    " :ref:`sequence_slice <cn_api_paddle_static_nn_sequence_slice>` ", "仅支持带有 LoD 信息的 Tensor，对输入的 Tensor，实现序列切片运算"
    " :ref:`sequence_softmax <cn_api_paddle_static_nn_sequence_softmax>` ", "仅支持带有 LoD 信息的 Tensor，根据 Tensor 信息将输入的第 0 维度进行划分，在划分的每一个区间内部进行运算"

.. _about_io:

io 相关 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`deserialize_persistables <cn_api_paddle_static_deserialize_persistables>` ", "反序列化模型参数"
    " :ref:`deserialize_program <cn_api_paddle_static_deserialize_program>` ", "反序列化 program"
    " :ref:`load <cn_api_paddle_static_load>` ", "加载模型"
    " :ref:`load_from_file <cn_api_paddle_static_load_from_file>` ", "从指定的文件中加载内容"
    " :ref:`load_inference_model <cn_api_paddle_static_load_inference_model>` ", "加载预测模型"
    " :ref:`load_program_state <cn_api_paddle_static_load_program_state>` ", "加载 Program 的参数与优化器信息"
    " :ref:`save <cn_api_paddle_static_save>` ", "保存模型"
    " :ref:`save_inference_model <cn_api_paddle_static_save_inference_model>` ", "保存预测模型"
    " :ref:`save_to_file <cn_api_paddle_static_save_to_file>` ", "将内容写入指定的文件"
    " :ref:`serialize_persistables <cn_api_paddle_static_serialize_persistables>` ", "序列化模型参数"
    " :ref:`serialize_program <cn_api_paddle_static_serialize_program>` ", "序列化 program"

.. _about_variable:

变量相关 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`create_global_var <cn_api_paddle_static_create_global_var>` ", "创建全局变量"
    " :ref:`data <cn_api_paddle_static_data>` ", "在全局 block 中创建变量"
    " :ref:`gradients <cn_api_paddle_static_gradients>` ", "将目标变量的梯度反向传播到输入变量"
    " :ref:`Print <cn_api_paddle_static_Print>` ", "打印正在访问的变量内容"
    " :ref:`Variable <cn_api_paddle_static_Variable>` ", "创建参数"
    " :ref:`WeightNormParamAttr <cn_api_paddle_static_WeightNormParamAttr>` ", "权重归一化类"
    " :ref:`sequence_scatter <cn_api_paddle_static_nn_sequence_scatter>` ", "仅支持 LoDTensor,根据 index 提供的位置将 updates 中的信息更新到输出中"
    " :ref:`sequence_unpad <cn_api_paddle_static_nn_sequence_unpad>` ", "仅支持 LoDTensor ，根据 length 的信息，将 input 中 padding 元素移除，并且返回一个 LoDTensor"
.. _about_device:

运行设备相关 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`cpu_places <cn_api_paddle_static_cpu_places>` ", "创建 `paddle.CPUPlace` 对象"
    " :ref:`cuda_places <cn_api_paddle_static_cuda_places>` ", "创建 `paddle.CUDAPlace` 对象"
    " :ref:`device_guard <cn_api_paddle_static_device_guard>` ", "用于指定 OP 运行设备的上下文管理器"

.. _about_metrics:

评估指标相关 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`accuracy <cn_api_paddle_static_accuracy>` ", "计算精确率"
    " :ref:`auc <cn_api_paddle_static_auc>` ", "计算 AUC"


.. _about_others:

其他 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`global_scope <cn_api_paddle_static_global_scope>` ", "获取全局/默认作用域实例"
    " :ref:`InputSpec <cn_api_paddle_static_InputSpec>` ", "描述模型输入的签名信息"
    " :ref:`name_scope <cn_api_paddle_static_py_func>` ", "为 OP 生成命名空间"
    " :ref:`py_func <cn_api_paddle_static_py_func>` ", "自定义算子"
    " :ref:`scope_guard <cn_api_paddle_static_scope_guard>` ", "切换作用域"
    " :ref:`while_loop <cn_api_paddle_static_nn_while_loop>` ", "while 循环控制"
