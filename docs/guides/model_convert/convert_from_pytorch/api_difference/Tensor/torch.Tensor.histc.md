## [ torch 参数更多 ]torch.Tensor.histc

### [torch.Tensor.histc](https://pytorch.org/docs/1.13/generated/torch.Tensor.histc.html?highlight=torch+tensor+histc#torch.Tensor.histc)

```python
torch.Tensor.histc(input, bins=100, min=0, max=0, *, out=None)
```

### [paddle.histogram](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/histogram_cn.html#histogram)

```python
paddle.histogram(input, bins=100, min=0, max=0, name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch                           | PaddlePaddle                 | 备注                                                   |
|-----------------------------------|------------------------------| ------------------------------------------------------ |
| <font color='red'> out </font>    | -                            | 表示输出的 Tensor，PaddlePaddle 无此参数，需要进行转写。              |

### 转写示例

#### out：指定输出
```python
# Pytorch 写法
torch.Tensor.histc(torch.tensor([1., 2, 1]), bins=4, min=0, max=3, out = y)

# Paddle 写法
y = paddle.histogram(paddle.to_tensor([1, 2, 1]), bins=4, min=0, max=3)
