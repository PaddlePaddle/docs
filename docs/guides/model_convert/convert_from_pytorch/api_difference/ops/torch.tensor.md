## torch.tensor
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
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| device        | place        | 表示 Tensor 存放位置。                   |
| requires_grad | stop_gradient| PyTorch 表示是否不阻断梯度传导，PaddlePaddle 表示是否阻断梯度传导。 |
| pin_memeory   | -            | 表示是否使用锁页内存，PaddlePaddle 无此参数。           |


### 转写示例
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
