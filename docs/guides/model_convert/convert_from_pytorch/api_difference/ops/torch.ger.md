## [torch 参数更多 ]torch.ger

### [torch.ger](https://pytorch.org/docs/1.13/generated/torch.ger.html?highlight=ger#torch.ger)

```python
torch.ger(input, vec2, *, out=None)
```

### [paddle.outer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/outer_cn.html)

```python
paddle.outer(x, y, name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font>         | <font color='red'> x </font>            | 输入的 Tensor ，仅参数名不同。                                     |
| <font color='red'> vec2 </font>         | <font color='red'> y </font>            | 输入的 Tensor ，仅参数名不同。                                     |
| <font color='red'> out </font>           | -                                       | 表示输出的 Tensor，PaddlePaddle 无此参数，需要进行转写。              |


### 转写示例

#### out：指定输出
```python
# Pytorch 写法
torch.ger(torch.arange(1., 5.), torch.arange(1., 4.), out=y)

# Paddle 写法
paddle.assign(paddle.outer(paddle.arange(1, 5).astype('float32'), paddle.arange(1, 4).astype('float32')), y)
```
