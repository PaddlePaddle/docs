## [torch 参数更多]torch.special.gammaln

### [torch.special.gammaln](https://pytorch.org/docs/stable/special.html#torch.special.gammaln)

```python
torch.special.gammaln(input, *, out=None)
```

### [paddle.lgamma](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/lgamma_cn.html#lgamma)

```python
paddle.lgamma(x, name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                 |
| ------- | ------------ | ---------------------------------------------------- |
| input   | x            | 输入的 Tensor，仅参数名不一致。                      |
| out     | -            | 表示输出的 Tensor ， Paddle 无此参数，需要转写。 |

### 转写示例

#### out：指定输出

```python
# Pytorch 写法
torch.special.gammaln(x, out=y)

# Paddle 写法
paddle.assign(paddle.lgamma(x), y)
```
