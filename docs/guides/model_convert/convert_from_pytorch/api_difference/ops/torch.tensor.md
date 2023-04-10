## [torch 参数更多 ]torch.tensor
### [torch.tensor](https://pytorch.org/docs/stable/generated/torch.tensor.html?highlight=tensor#torch.tensor)

```python
torch.tensor(data,
             dtype=None,
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

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| data        | data        | 表示输入的数据。                   |
| dtype        | dtype        | 表示数据的类型。                   |
| <font color='red'> device </font>     | <font color='red'> place </font>       | 表示 Tensor 存放设备位置，两者参数类型不相同，需要进行转写。 |
| <font color='red'> requires_grad </font> | <font color='red'> stop_gradient </font>   | 表示是否计算梯度， 两者参数意义不相同，需要进行转写。 |
| <font color='red'> pin_memeory </font>   | - | 表示是否使用锁页内存， Paddle 无此参数，需要进行转写。   |

### 转写示例

#### device: Tensor 的设备
```python
# Pytorch 写法
torch.tensor(3, device=torch.device('cpu'))

# Paddle 写法
y = paddle.to_tensor(3, place=paddle.CPUPlace())
```

#### requires_grad：是否不阻断梯度传导
```python
# Pytorch 写法
x = torch.tensor(3, requires_grad=True)

# Paddle 写法
x = paddle.to_tensor(3, stop_gradient=False)
```

#### pin_memory：是否分配到固定内存上
```python
# Pytorch 写法
x = torch.tensor(3, pin_memory=True)

# Paddle 写法
x = paddle.to_tensor(3).pin_memory()
```
