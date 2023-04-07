##  [ 无参数 ] torch.Tensor.requires_grad

### [torch.Tensor.requires_grad](https://pytorch.org/docs/2.0/generated/torch.Tensor.requires_grad.html#torch.Tensor.requires_grad)

```python
torch.Tensor.requires_grad
```

### [paddle.Tensor.stop_gradient](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#stop-gradient)

```python
paddle.Tensor.stop_gradient
```

两者功能一致，均无参数，表示 Tensor 是否停止计算梯度 。

### 转写示例
#### Tensor.requires_grad 作为右值
```python
# torch 写法
y = x.requires_grad

# paddle 写法
y = not x.stop_gradient
```
#### Tensor.requires_grad 作为左值
```python
# torch 写法
x.requires_grad = True

# paddle 写法
x.stop_gradient = not True
