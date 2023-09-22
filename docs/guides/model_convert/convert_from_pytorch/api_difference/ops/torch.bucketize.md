## [torch 参数更多 ] torch.bucketize
### [torch.bucketize](https://pytorch.org/docs/stable/generated/torch.bucketize.html#torch.bucketize)

```python
torch.bucketize(input, boundaries, *, out_int32=False, right=False, out=None)
```

### [paddle.bucketize](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/bucketize_cn.html#paddle-bucketize)

```python
paddle.bucketize(x, sorted_sequence, out_int32=False, right=False, name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'>input</font>| <font color='red'>x</font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'>boundaries</font>| <font color='red'>sorted_sequence</font> | 数据的边界，仅参数名不一致。  |
| <font color='red'>out_int32</font>| <font color='red'>out_int32</font> | 输出的数据类型是否为 int32。  |
| <font color='red'>right</font>| <font color='red'>right</font> | 根据给定输入在 sorted_sequence 查找对应的上边界或下边界。  |
| <font color='red'>out</font> | -  | 表示输出的 Tensor， Paddle 无此参数，需要转写。    |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.bucketize(x, boundaries, out=y)

# Paddle 写法
paddle.assign(paddle.bucketize(x, boundaries), y)
```
