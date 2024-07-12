## [torch 参数更多]fairscale.nn.model_parallel.layers.ParallelEmbedding

### [fairscale.nn.model_parallel.layers.ParallelEmbedding](https://github.com/facebookresearch/fairscale/blob/164cc0f3170b4a3951dd84dda29c3e1504ac4d6e/fairscale/nn/model_parallel/layers.py#L152)

```python
fairscale.nn.model_parallel.initialize.ParallelEmbedding(num_embeddings,embedding_dim,padding_idx,max_norm,norm_type,scale_grad_by_freq,sparse,init_method,keep_master_weight_for_test)
```
### [paddle.distributed.meta_parallel.parallel_layers.mp_layers.VocabParallelEmbedding](https://github.com/PaddlePaddle/Paddle/blob/016766cc89fabc10181453ce70b701dd8ed019f6/python/paddle/distributed/fleet/layers/mpu/mp_layers.py#L37)

```python
paddle.distributed.meta_parallel.parallel_layers.mp_layers.VocabParallelEmbedding(num_embeddings,embedding_dim,weight_attr,mp_group,name)
```

两者功能大体一致，但内部实现细节不一样，ParallelEmbedding 的切分方向沿着 embedding 方向，VocabParallelEmbedding 的切分方向沿着 vocab(词汇表)方向。

### 参数映射

| fairscale | PaddlePaddle | 备注     |
| --------- | ------------ | -------- |
| num_embeddings | num_embeddings|词汇表大小 |
| embedding_dim |embedding_dim |embedding 的维度大小|
| padding_idx | | 填充下标处的数据对梯度无贡献 |
| max_norm | | 范数大于 maxnorm 的数值被设置为 maxnorm|
| norm_type | | 设置 p 范数|
| sparse | | 是否为稀疏向量 |
| scale_grad_by_freq| | 是否根据 batch 内单词的频数的倒数缩放梯度 |
| init_method | | 参数初始化方法|
| keep_master_weight_for_test | | 返回主参数用于测试 |
|  | mp_group| 模型并行组|
|  | name| 网络层名称|


### 转写示例

```python
# Pytorch 写法
fairscale.nn.model_parallel.initialize.ParallelEmbedding(num_embeddings=num_embeddings,
    embedding_dim=embedding_dim)

# Paddle 写法
paddle.distributed.meta_parallel.parallel_layers.mp_layers.VocabParallelEmbedding(num_embeddings=num_embeddings,
    embedding_dim=embedding_dim)

```
