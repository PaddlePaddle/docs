## [ torch 参数更多 ]torch.nn.Embedding
### [torch.nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html?highlight=embedding#torch.nn.Embedding)

```python
torch.nn.Embedding(num_embeddings,
                   embedding_dim,
                   padding_idx=None,
                   max_norm=None,
                   norm_type=2.0,
                   scale_grad_by_freq=False,
                   sparse=False)
```
### [paddle.nn.Embedding](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Embedding_cn.html#embedding)

```python
paddle.nn.Embedding(num_embeddings,
                    embedding_dim,
                    padding_idx=None,
                    sparse=False,
                    weight_attr=None,
                    name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| num_embeddings     | num_embeddings            | 表示嵌入字典的大小。  |
| embedding_dim     | embedding_dim            | 表示每个嵌入向量的维度。  |
| padding_idx     | padding_idx            | 在此区间内的参数及对应的梯度将会以 0 进行填充  |
| max_norm      | -            | 如果给定，Embeddding 向量的范数（范数的计算方式由 norm_type 决定）超过了 max_norm 这个界限，就要再进行归一化，Paddle 无此参数，暂无转写方式。  |
| norm_type     | -            | 为 maxnorm 选项计算 p-范数的 p。默认值 2，Paddle 无此参数，暂无转写方式。  |
| scale_grad_by_freq | -       | 是否根据单词在 mini-batch 中出现的频率，对梯度进行放缩，Paddle 无此参数，暂无转写方式。  |
| sparse     | sparse            | 表示是否使用稀疏更新。  |
| -             | weight_attr  | 指定权重参数属性的对象，Pytorch 无此参数，Paddle 保持默认即可。  |
