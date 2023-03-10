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

其中 Pytorch 的 count_include_pad 与 Paddle 的 exclusive 用法不一致，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| max_norm      | -            | 如果给定，Embeddding 向量的范数（范数的计算方式由 norm_type 决定）超过了 max_norm 这个界限，就要再进行归一化，PaddlePaddle 无此功能，暂无转写方式。  |
| norm_type     | -            | 为 maxnorm 选项计算 p-范数的 p。默认值 2，PaddlePaddle 无此功能，暂无转写方式。  |
| scale_grad_by_freq | -       | 是否根据单词在 mini-batch 中出现的频率，对梯度进行放缩，PaddlePaddle 无此功能。  |
| -             | weight_attr  | 指定权重参数属性的对象，Pytorch 无此参数，Paddle 保持默认即可。  |
