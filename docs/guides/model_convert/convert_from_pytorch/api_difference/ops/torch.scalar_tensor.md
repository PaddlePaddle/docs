## [torch 参数更多 ]torch.scalar_tensor
### [torch.scalar_tensor]

```python
torch.scalar_tensor(s,
             dtype=torch.float32,
             layout=torch.strided,
             device=None,
             requires_grad=False,
             pin_memory=False)
```

### [paddle.to_tensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/to_tensor_cn.html#to-tensor)

```python
paddle.to_tensor(data,
                 dtype=None,
                 place=None,
                 stop_gradient=True)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| s        | data        | 表示输入的数据。                   |
| dtype        | dtype        | 表示数据的类型。两者参数默认值不同，Paddle 应设置为 paddle.float32。                  |
|  layout    |  -        | 数据布局格式，Paddle 无此参数。一般对训练结果影响不大，可直接删除。 |
|  device    |  place        | 表示 Tensor 存放设备位置，两者参数类型不相同，需要转写。 |
|  requires_grad  |  stop_gradient   | 表示是否计算梯度， 两者参数意义不相同，Paddle 输入与 Pytorch 逻辑相反。需要转写。 |
|  pin_memeory    | - | 表示是否使用锁页内存， Paddle 无此参数，需要转写。 Paddle 需要对结果使用 padlde.Tensor.pin_memory()。  |

### 转写示例
#### device: Tensor 的设备
```python
# Pytorch 写法
torch.tensor(3, device=torch.device('cpu'))

# Paddle 写法
y = paddle.to_tensor(3, dtype=paddle.float32, place=paddle.CPUPlace())
```

#### requires_grad：是否不阻断梯度传导
```python
# Pytorch 写法
x = torch.tensor(3, requires_grad=True)

# Paddle 写法
x = paddle.to_tensor(3, dtype=paddle.float32, stop_gradient=False)
```

#### pin_memory：是否分配到固定内存上
```python
# Pytorch 写法
x = torch.tensor(3, pin_memory=True)

# Paddle 写法
x = paddle.to_tensor(3, dtype=paddle.float32).pin_memory()
```
