## [ 仅参数名不一致 ]torch.permute
### [torch.permute](https://pytorch.org/docs/stable/generated/torch.permute.html?highlight=permute#torch.permute)

```python
torch.permute(input,
              dims)
```

### [paddle.transpose](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/transpose_cn.html#transpose)

```python
paddle.transpose(x,
                 perm,
                 name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的多维 Tensor，仅参数名不一致。                   |
| dims          | perm         | dims 长度必须和 input 的维度相同，并依照 dims 中数据进行重排，仅参数名不一致。 |
