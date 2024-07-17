## [torch 参数更多]fairscale.nn.model_parallel.layers.ParallelEmbedding

### [fairscale.nn.model_parallel.layers.ParallelEmbedding](https://github.com/facebookresearch/fairscale/blob/164cc0f3170b4a3951dd84dda29c3e1504ac4d6e/fairscale/nn/model_parallel/layers.py#L152)

```python
fairscale.nn.model_parallel.initialize.ParallelEmbedding(num_embeddings: int, embedding_dim: int ,padding_idx: Optional[int] = None, max_norm: Optional[float] = None, norm_type: float = 2.0, scale_grad_by_freq: bool = False, sparse: bool = False, init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_, keep_master_weight_for_test: bool = False)
```
### [paddle.distributed.meta_parallel.parallel_layers.mp_layers.VocabParallelEmbedding](https://github.com/PaddlePaddle/Paddle/blob/016766cc89fabc10181453ce70b701dd8ed019f6/python/paddle/distributed/fleet/layers/mpu/mp_layers.py#L37)

```python
paddle.distributed.meta_parallel.parallel_layers.mp_layers.VocabParallelEmbedding(num_embeddings, embedding_dim, weight_attr=None, mp_group=None, name=None)
```

两者功能大体一致，但内部实现细节不一样，ParallelEmbedding 的切分方向沿着 embedding 方向，VocabParallelEmbedding 的切分方向沿着 vocab(词汇表)方向，故在多卡训练时，load 参数时需手动修改以匹配参数切分方式的不同。

### 参数映射

| fairscale                    | PaddlePaddle   | 备注      |
| ---------------------------- | -------------- | -------- |
| num_embeddings               | num_embeddings | 词汇表大小  |
| embedding_dim                | embedding_dim  | embedding 的维度大小|
| padding_idx                  | -              | 填充数值，Paddle 无此参数，暂无转写方式 |
| max_norm                     | -              | 范数大于 maxnorm 的数值被设置为 maxnorm |
| norm_type                    | -              | 设置 p 范数，Paddle 无此参数，暂无转写方式 |
| sparse                       | -              | 是否为稀疏向量，Paddle 无此参数，暂无转写方式 |
| scale_grad_by_freq           | -              | 是否根据 batch 内单词的频数的倒数缩放梯度，Paddle 无此参数，暂无转写方式|
| init_method                  | -              | 参数初始化方法，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除 |
| keep_master_weight_for_test  | -              | 返回主参数用于测试，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除 |
| -                            | mp_group       | 模型并行组 |
| -                            | name           | 网络层名称 |
