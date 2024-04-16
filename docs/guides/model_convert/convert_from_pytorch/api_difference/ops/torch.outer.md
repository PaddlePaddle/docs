## [torch 参数更多 ]torch.outer

### [torch.outer](https://pytorch.org/docs/stable/generated/torch.outer.html#torch.outer)

```python
torch.outer(input, vec2, *, out=None)
```

### [paddle.outer](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/outer_cn.html)

```python
paddle.outer(x, y, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font>         | <font color='red'> x </font>            | 输入的 Tensor ，仅参数名不一致。                                     |
| <font color='red'> vec2 </font>         | <font color='red'> y </font>            | 输入的 Tensor ，仅参数名不一致。                                     |
| <font color='red'> out </font>           | -                                       | 表示输出的 Tensor，Paddle 无此参数，需要转写。              |


### 转写示例

#### out：指定输出
```python
# PyTorch 写法
torch.outer([1., 2., 3., 4.], [1., 2., 3.], out=y)

# Paddle 写法
paddle.assign(paddle.outer([1., 2., 3., 4.], [1., 2., 3.]), y)
```
