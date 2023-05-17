## [torch 参数更多 ]torch.bucketize

### [torch.bucketize](https://pytorch.org/docs/stable/generated/torch.bucketize.html?highlight=bucketize#torch.bucketize)

```python
torch.bucketize(input,
                boundaries,
                *,
                out_int32=False,
                right=False,
                out=None)
```

### [paddle.bucketize](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/bucketize_cn.html#paddle.bucketize)

```python
paddle.bucketize(x,
                 sorted_sequence,
                 out_int32=False,
                 right=False,
                 name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 N 维 Tensor ，仅参数名不同。                       |
| boundaries    | sorted_sequence   | 输入的一维 Tensor，该 Tensor 的值在其最后一个维度递增，仅参数名不同。   |
| out_int32     | out_int32    | 输出的数据类型支持 int32、int64。默认值为 False，表示默认的输出数据类型为 int64，参数名相同。 |
| right         |right         | 根据给定 x 在 sorted_sequence 查找对应的上边界或下边界。如果 sorted_sequence 的值为 nan 或 inf，则返回最内层维度的大小。默认值为 False，参数名相同。   |
| out           | -            | 表示输出的 Tensor，必须和输入 input 的 size 相同。Paddle 无此参数，需要进行转写。      |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.bucketize(input, other, right=True, out=y)

# Paddle 写法
paddle.assign(paddle.bucketize(input, other, right=True), y)
```
