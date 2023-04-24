## [torch 参数更多 ]torch.maximum

### [torch.maximum](https://pytorch.org/docs/stable/generated/torch.maximum.html#torch.maximum)

```python
torch.maximum(input, other, *, out=None)
```

### [paddle.maximum](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/maximum_cn.html)

```python
paddle.maximum(x, y, name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。数据类型为 float32 、 float64 、 int32 或 int64。|
| other         | y            | 输入的 Tensor。数据类型为 float32 、 float64 、 int32 或 int64。  |
| out           | -            | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。    |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.maximum([3, 5], 1., out=y)

# Paddle 写法
y = paddle.maximum([3, 5], 1.)
```
