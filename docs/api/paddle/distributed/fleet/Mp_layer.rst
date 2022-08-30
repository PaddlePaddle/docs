Mp_layers
-----------------


..Fleet:mp_layers

Mp_layers是飞桨的分布式训练统一API中的fleet中的一个方法,其主要分为以下几个部分：嵌入式表示， 矩阵乘

嵌入式表示：

方法
........
........
init(num_embeddings, embedding_dim,  weight_attr=None, mp_group=None, name=None)
''''''''

初始化嵌入式类。

**参数**

    - **num_embeddings** (int) 词典的大小尺寸
    - **embedding_dim** (Tensor) 词典中每个词的向量维度
    - **weight_attr** (Tenosr) 指定权重参数的属性
    - **mp_group** (bool)    表示是否采用模型并行的集群
    - **name** (string)     神经网络模型输出的前缀标识

**返回**
None


create_parameter(weight_attr, shape, dtype, is_bias)
''''''''

创建参数模型的权重参数


**参数**

   - **weight_attr** (Tensor) 指定权重参数的属性
   - **shape** (Tensor)模型参数的形状（请注意这里的形状是由词典的大小尺寸整除以GPU数目与embedding_dim的维度连接而成）
   - **dtype** 参数的数据类型
   - **is_bias** (bool)  是否存在偏置项,以简单的W=AX+B为例，此处的B即为偏置项
**返回**
Tensor

**代码示例**
    code-block::python
         self.weight = self.create_parameter(attr=self._weight_attr,
                                                    shape=self._size,
                                                    dtype=self._dtype,
                                                    is_bias=False)



c_lookup_table（table, index, start_index, name）
''''''''

根据序列号查找表

**参数**

    - **table** (Tensor) 输入的张量矩阵
    - **index** (Tensor) 查找表的序列号
    - **start_index**  (int) 查重表的初始位置
    - **name** (string)   调用的api名称
**返回**
Tensor

**代码示例**
  code-block::python
       output_parallel = paddle.distributed.collective._c_lookup_table(
                self.weight,
                x,
                start_index=self.vocab_start_index,
                name=self._name)



mp_allreduce (output, group, use_cal_stram, use_model_parallel)
..........

对矩阵张量进行聚合

**参数**
    - **output** (Tensor) 输入的张量矩阵
    - **group** (int) 分布式并行的群组节点ID
    - **use_cal_stream** （bool）是否使用流水线
    - **use_model_parallel** （bool）是否采用模型并行
**返回**
Tensor

**代码示例**
   code-block::python
          output_parallel = paddle.distributed.collective._c_lookup_table(
                self.weight,
                x,
                start_index=self.vocab_start_index,
                name=self._name)


embedding (input, weight, padding_index, sparse, name)
''''''''''''

对张量进行嵌入式运行

**参数**
       - **input** (Tensor) 输入的张量
       - **weight** (Tensor) 模型权重参数
       - **padding_index** (int) 该位置下的向量用0补齐
       - **sparse** (bool) 在前向传播和后向传播的过程中不考虑全为0的向量
       - **name** (string) 张量名称等
**返回**
Tensor


矩阵乘：

方法
........
........
init(in_feature, out_feature,  weight_attr=None, has_bias, gather_output,  fuse_matmul_bias, mp_group, name)
''''''''

对矩阵分片运算进行初始化

**参数**
       - **in_feature** (Tensor) 输入的张量
       - **out_feature** (Tensor) 输出的张量
       - **weight_attr** (Tensor) 指定权重参数的属性
       - **has_bias** (bool) 是否拥有偏置项
       - **gater_output** (bool) 是否将本地结果进行聚合
       - **fuse_matmul_bias** (bool) 分片向量进行融合时是否存在偏置
       - **mp_group** (bool)    表示是否采用模型并行的集群
       - **name** (string) 张量名称等


c_concat(x, group)
'''''''

对分片后的张量进行切片

**参数**
       - **x** 输入的张量
       - **group** (int) 分布式并行的群组节点ID
**返回**
Tensor


**代码示例**
   code-block::python
                  output = paddle.distributed.collective._c_concat(
                output_parallel, group=self.model_parallel_group)


c_split(x, group)
'''''''

对分片后的张量进行连接

**参数**
       - **x** (Tensor) 输入的张量
       - **group** (int) 分布式并行的群组节点ID
**返回**
Tensor

**代码示例**
   code-block::python
           input_parallel = paddle.distributed.collective._c_split(
                x, group=self.model_parallel_group)


