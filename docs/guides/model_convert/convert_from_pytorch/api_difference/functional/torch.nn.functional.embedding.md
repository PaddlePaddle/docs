## [ 参数不一致 ]torch.nn.functional.embedding

### [torch.nn.functional.embedding](https://pytorch.org/docs/stable/generated/torch.nn.functional.embedding.html)

```python
torch.nn.functional.embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False)
```

### [paddle.nn.functional.embedding](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/embedding_cn.html#embedding)

```python
paddle.nn.functional.embedding(x, weight, padding_idx=None, sparse=False, name=None)
```

其中 PyTorch 和 Paddle 功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch            | PaddlePaddle       | 备注 |
| ------------------ | ------------------ | -- |
| input              | x                  | 输入 Tensor，仅参数名不同。   |
| weight             | weight             | 嵌入矩阵权重。                |
| padding_idx        | padding_idx        | 视为填充的下标，参数完全一致。 |
| max_norm           | -                  | 重新归一化的最大范数，参数不一致，暂无转写方式。        |
| norm_type          | -                  | Paddle 无此参数，参数不一致，暂无转写方式。                  |
| scale_grad_by_freq | -                  | 按词频进行梯度缩放的比例，参数不一致，暂无转写方式。 |
| sparse             | sparse             | 是否使用稀疏更新。            |
