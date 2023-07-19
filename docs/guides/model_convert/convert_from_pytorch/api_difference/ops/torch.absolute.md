## [torch 参数更多 ]torch.absolute

### [torch.absolute](https://pytorch.org/docs/stable/generated/torch.absolute.html?highlight=absolute#torch.absolute)

```python
torch.absolute(input,
               *,
               out=None)
```

### [paddle.abs](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/abs_cn.html#abs)

```python
paddle.abs(x,
           name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                     |
| ------- | ------------ | -------------------------------------------------------- |
| input   | x            | 表示输入的 Tensor ，仅参数名不一致。                     |
| out     | -            | 表示输出的 Tensor，PaddlePaddle 无此参数，需要进行转写。 |


### 转写示例

#### out：指定输出

```python
# Pytorch 写法
torch.absolute(torch.tensor([-3, -5]), out=y)

# Paddle 写法
paddle.assign(paddle.abs(paddle.to_tensor([-3, -5])), y)
```
