## [ 仅参数名不一致 ]torch.unsqueeze
### [torch.unsqueeze](https://pytorch.org/docs/stable/generated/torch.unsqueeze.html?highlight=unsqueeze#torch.unsqueeze)

```python
torch.unsqueeze(input,
                dim)
```

### [paddle.unsqueeze](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/unsqueeze_cn.html#unsqueeze)

```python
paddle.unsqueeze(x,
                 axis,
                 name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor ，仅参数名不一致。                   |
| dim           | axis         | 表示要插入维度的位置，仅参数名不一致。 |
