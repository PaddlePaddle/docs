## [ torch 参数更多 ]torch.nn.functional.embedding
### [torch.nn.functional.embedding](https://pytorch.org/docs/stable/generated/torch.nn.functional.embedding.html?highlight=torch+nn+functional+embedding#torch.nn.functional.embedding)

```python
torch.nn.functional.embedding(input,
                              weight,
                              padding_idx=None,
                              max_norm=None,
                              norm_type=2.0,
                              scale_grad_by_freq=False,
                              sparse=False)
```
### [paddle.nn.functional.embedding](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/embedding_cn.html#embedding)

```python
paddle.nn.functional.embedding(x,
                               weight,
                               padding_idx=None,
                               sparse=False,
                               name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input     | x            | 表示存储id信息的Tensor  |
| weight     | weight            | 表示存储词嵌入权重参数的 Tensor  |
| padding_idx     | padding_idx            | 在此区间内的参数及对应的梯度将会以 0 进行填充  |
| max_norm      | -            | 如果给定，Embeddding 向量的范数（范数的计算方式由 norm_type 决定）超过了 max_norm 这个界限，就要再进行归一化，PaddlePaddle 无此功能，暂无转写方式。  |
| norm_type     | -            | 为 maxnorm 选项计算 p-范数的 p。默认值 2，PaddlePaddle 暂无此功能，暂无转写方式。  |
| scale_grad_by_freq | -       | 是否根据单词在 mini-batch 中出现的频率，对梯度进行放缩，PaddlePaddle 暂无此功能。  |
| sparse     | sparse            | 表示是否使用稀疏更新。  |
| -             | name  | Pytorch 无此参数，Paddle 保持默认即可。  |
