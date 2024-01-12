## [torch 参数更多 ]torch.arccosh

### [torch.arccosh](https://pytorch.org/docs/stable/generated/torch.arccosh.html?highlight=arccosh#torch.arccosh)

```python
torch.arccosh(input,
             *,
             out=None)
```

### [paddle.acosh](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/acosh_cn.html#acos)

```python
paddle.acosh(x,
            name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                      |
| ------- | ------------ | --------------------------------------------------------- |
| input   | x            | 表示输入的 Tensor ，仅参数名不一致。                      |
| out     | -            | 表示输出的 Tensor ，Paddle 无此参数，需要转写。 |


### 转写示例

#### out：指定输出

```python
# PyTorch 写法
torch.arccosh(torch.tensor([1.3192, 1.9915]), out=y)

# Paddle 写法
paddle.assign(paddle.acosh(paddle.to_tensor([1.3192, 1.9915])), y)
```
