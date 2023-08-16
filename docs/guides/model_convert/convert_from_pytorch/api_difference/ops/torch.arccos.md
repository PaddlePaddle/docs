## [torch 参数更多 ]torch.arccos

### [torch.arccos](https://pytorch.org/docs/stable/generated/torch.arccos.html?highlight=arccos#torch.arccos)

```python
torch.arccos(input,
             *,
             out=None)
```

### [paddle.acos](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/acos_cn.html#acos)

```python
paddle.acos(x,
            name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                      |
| ------- | ------------ | --------------------------------------------------------- |
| input   | x            | 表示输入的 Tensor ，仅参数名不一致。                      |
| out     | -            | 表示输出的 Tensor ，Paddle 无此参数，需要转写。 |


### 转写示例

#### out：指定输出

```python
# Pytorch 写法
torch.arccos(torch.tensor([0.3348, -0.5889]), out=y)

# Paddle 写法
paddle.assign(paddle.acos(paddle.to_tensor([0.3348, -0.5889])), y)
```
