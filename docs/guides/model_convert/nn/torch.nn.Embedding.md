## torch.nn.Embedding
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
### [paddle.nn.Embedding](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Embedding_cn.html#embedding)
```python
paddle.nn.Embedding(num_embeddings,
                    embedding_dim,
                    padding_idx=None,
                    sparse=False,
                    weight_attr=None,
                    name=None)
```

### 功能差异
#### 归一化设置
***PyTorch***：当 max_norm 不为`None`时，如果 Embeddding 向量的范数（范数的计算方式由 norm_type 决定）超过了 max_norm 这个界限，就要再进行归一化。
***PaddlePaddle***：PaddlePaddle 无此要求，因此不需要归一化。

#### 梯度缩放设置
***PyTorch***：若 scale_grad_by_freq 设置为`True`，会根据单词在 mini-batch 中出现的频率，对梯度进行放缩。
***PaddlePaddle***：PaddlePaddle 无此功能。
