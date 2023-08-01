## [ 仅参数名不一致 ]torch.nn.Softmax
### [torch.nn.Softmax](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html?highlight=nn+softmax#torch.nn.Softmax)

```python
torch.nn.Softmax(dim=None)
```

### [paddle.nn.Softmax](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Softmax_cn.html#softmax)

```python
paddle.nn.Softmax(axis=- 1,
                  name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| dim           | axis         | 指定对输入 Tensor 进行运算的轴，仅参数名不一致。                          |
