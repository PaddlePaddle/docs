## [torch 参数更多 ]torch.inverse

### [torch.inverse](https://pytorch.org/docs/stable/generated/torch.inverse.html?highlight=inverse#torch.inverse)

```python
torch.inverse(input, *, out=None)
```

### [paddle.linalg.inv](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/inv_cn.html)

```python
paddle.linalg.inv(x, name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font>         | <font color='red'> x </font>            | 输入的 Tensor ，仅参数名不一致。                                     |
| <font color='red'> out </font>           | -                                       | 表示输出的 Tensor，Paddle 无此参数，需要转写。              |


### 转写示例

#### out：指定输出
```python
# Pytorch 写法
torch.inverse(torch.tensor([[2., 0.], [0., 2.]]), out=y)

# Paddle 写法
paddle.assign(paddle.inverse(paddle.to_tensor([[2., 0], [0, 2.]])), y)
```
