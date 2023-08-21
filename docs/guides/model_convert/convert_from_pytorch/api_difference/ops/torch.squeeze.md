## [ 仅参数名不一致 ]torch.squeeze
### [torch.squeeze](https://pytorch.org/docs/stable/generated/torch.squeeze.html?highlight=squeeze#torch.squeeze)

```python
torch.squeeze(input,
              dim=None)
```

### [paddle.squeeze](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/squeeze_cn.html#squeeze)

```python
paddle.squeeze(x,
               axis=None,
               name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> dim </font> | <font color='red'> axis </font> | 表示要压缩的轴，仅参数名不一致。  |
