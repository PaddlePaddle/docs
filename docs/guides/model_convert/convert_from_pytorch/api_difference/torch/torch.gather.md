## [torch 参数更多 ]torch.gather
### [torch.gather](https://pytorch.org/docs/stable/generated/torch.gather.html?highlight=gather#torch.gather)

```python
torch.gather(input,
             dim,
             index,
             *,
             sparse_grad=False,
             out=None)
```

### [paddle.gather](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/gather_cn.html#gather)

```python
paddle.gather(input,
              dim,
              index,
              out=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | input        | 表示输入 Tensor ，参数名保持一致。                                    |
| dim           | dim          | 用于指定 index 获取输入的维度，参数名保持一致。                         |
| index         | index        | 聚合元素的索引矩阵，维度和输入 (input) 的维度一致，参数名保持一致。          |
| sparse_grad   | -            | 表示是否对梯度稀疏化，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。    |
| out           | out          | 表示目标 Tensor ，用于引用式返回结果，保持一致。         |


### 转写示例
#### out：指定输出
``` python
# PyTorch 写法：
t = torch.tensor([[1, 2], [3, 4]])
torch.gather(t, dim = 1, index = torch.tensor([[0, 0], [1, 0]]), out = y)

# Paddle 写法：
t = paddle.to_tensor([[1, 2], [3, 4]])
paddle.gather(t, dim = 1, index = paddle.to_tensor([[0, 0], [1, 0]]), out = y)
```
