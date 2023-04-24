## [torch 参数更多 ]torch.acosh

### [torch.acosh](https://pytorch.org/docs/1.13/generated/torch.acosh.html?highlight=acosh#torch.acosh)

```python
torch.arccos(input,
             *,
             out=None)
```

### [paddle.acosh](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/acosh_cn.html#acos)

```python
paddle.acos(x,
            name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                  |
| ------- | ------------ | ----------------------------------------------------- |
| input   | x            | 表示输入的 Tensor ，仅参数名不一致。                  |
| out     | -            | 表示输出的 Tensor ，PaddlePaddle 无此参数，需要转写。 |


### 转写示例

#### out：指定输出

```python
# Pytorch 写法
torch.acosh(torch.tensor([1.3192, 1.9915]), out=y)

# Paddle 写法
paddle.assign(paddle.acosh(paddle.to_tensor([1.3192, 1.9915])), y)
```
