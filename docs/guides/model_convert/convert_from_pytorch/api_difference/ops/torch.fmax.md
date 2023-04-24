## [torch 参数更多 ]torch.fmax

### [torch.fmax](https://pytorch.org/docs/stable/generated/torch.fmax.html#torch.fmax)

```python
torch.fmax(input, other, *, out=None)
```

### [paddle.fmax](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fmax_cn.html)

```python
paddle.fmax(x, y, name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。数据类型为 float16 、 float32 、 float64 、 int32 或 int64，仅参数名不一致。|
| other         | y            | 输入的 Tensor。数据类型为 float16 、 float32 、 float64 、 int32 或 int64，仅参数名不一致。|
| out           | -            | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。    |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.fmax([3, 5], 1., out=y)

# Paddle 写法
paddle.assign(paddle.fmax([3, 5], 1.), y)
```
