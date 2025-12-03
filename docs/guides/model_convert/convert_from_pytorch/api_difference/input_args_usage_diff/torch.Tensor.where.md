## [ 输入参数用法不一致 ]torch.Tensor.where
### [torch.Tensor.where](https://pytorch.org/docs/stable/generated/torch.Tensor.where.html#torch.Tensor.where)
```python
torch.Tensor.where(condition, other)
```

### [paddle.where](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/where_cn.html)
```python
paddle.where(condition, x=None, y=None, name=None)
```

Pytorch 为 Tensor 类方法，Paddle 为普通函数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| condition     | condition    | 判断条件。|
| self             | x            | 当 condition 为 true 时，选择的元素，调用 torch.Tensor 类方法的 self Tensor 传入。|
| other             | y            | 当 condition 为 false 时，选择的元素，仅参数名不一致。需要转写。|

### 转写示例
#### other 参数
```python
# torch 写法
c = a.where(a > 0, b)

# paddle 写法
c = paddle.where(a > 0, a, b)
```
