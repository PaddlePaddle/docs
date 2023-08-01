## [ 仅参数名不一致 ]torch.unbind
### [torch.unbind](https://pytorch.org/docs/stable/generated/torch.unbind.html?highlight=unbind#torch.unbind)

```python
torch.unbind(input,
             dim=0)
```

### [paddle.unbind](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/unbind_cn.html#unbind)

```python
paddle.unbind(x,
              axis=0)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的多维 Tensor ，仅参数名不一致。                   |
| dim           | axis         | 表示需要分割的维度，仅参数名不一致。 |
