## [组合替代实现]torch.Tensor.requires_grad_

### [torch.Tensor.requires_grad_](https://pytorch.org/docs/stable/generated/torch.Tensor.requires_grad_.html?highlight=requires_grad_#torch.Tensor.requires_grad_)

```python
torch.Tensor.requires_grad_(requires_grad=True)
```

### [paddle.Tensor.stop_gradient](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#stop-gradient)

```python
paddle.Tensor.stop_gradient = False
```

两者功能一致，torch 为 funtion 调用方式，paddle 为 attribution 赋值方式，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| requires_grad        | -            | 是否计算梯度，Paddle 无此参数，需要转写。                                      |


### 转写示例
#### requires_grad：是否计算梯度
```python
# 当 requires_grad 为‘True’时，torch 写法
x.requires_grad_(requires_grad=True)

# paddle 写法
x.stop_gradient = False

# 当 requires_grad 为‘False’时，torch 写法
x.requires_grad_(requires_grad=False)

# paddle 写法
x.stop_gradient = True
```
