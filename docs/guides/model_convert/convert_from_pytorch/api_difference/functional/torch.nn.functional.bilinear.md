## [ 参数不一致 ]torch.nn.functional.bilinear

### [torch.nn.functional.bilinear](https://pytorch.org/docs/stable/generated/torch.nn.functional.bilinear.html?highlight=bilinear#torch.nn.functional.bilinear)

```python
torch.nn.functional.bilinear(input1,
                             input2,
                             weight,
                             bias=None)
```

### [paddle.nn.functional.bilinear](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/bilinear_cn.html)

```python
paddle.nn.functional.bilinear(x1,
                              x2,
                              weight,
                              bias=None,
                              name=None)
```

两者功能一致，参数 `bias` 的 `shape` 不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input1          | x1         | 表示第一个输入的 Tensor ，仅参数名不一致。                                     |
| input2          | x2         | 表示第二个输入的 Tensor ，仅参数名不一致。                                     |
| weight          | weight         | 表示权重 Tensor 。                                     |
| bias          | bias         | 表示偏重 Tensor ，PyTorch bias 参数的 shape 为 (out_features)，Paddle bias 参数的 shape 为 (1, out_features)，需要转写。                                     |

### 转写示例
#### bias: 偏重
```python
# PyTorch 写法
result = F.bilinear(input1, input2, weight=weight, bias=bias)

# Paddle 写法
result = F.bilinear(input1, input2, weight=weight, bias=bias.unsqueeze(0))
```
