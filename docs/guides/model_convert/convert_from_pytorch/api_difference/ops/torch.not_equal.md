## [torch 参数更多 ]torch.not_equal

### [torch.not_equal](https://pytorch.org/docs/stable/generated/torch.not_equal.html?highlight=torch.not_equal#torch.not_equal)

```python
torch.not_equal(input,
         other,
         *,
         out=None)
```

### [paddle.not_equal](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/not_equal_cn.html#not_equal)

```python
paddle.not_equal(x,
                 y,
                 name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------|
| input         | x            | 输入的 Tensor ，仅参数名不一致。                          |
| other         | y            | 输入的 Tensor ，仅参数名不一致。                          |
| out           | -            | 表示输出的 Tensor ，Paddle 无此参数，需要转写。      |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.not_equal(input, other，out=y)

# Paddle 写法
paddle.assign(paddle.not_equal(input, other, y))
```
