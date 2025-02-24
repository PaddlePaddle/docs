## [组合替代实现]torch.Tensor.requires_grad_

### [torch.Tensor.requires_grad](https://pytorch.org/docs/stable/generated/torch.Tensor.requires_grad.html)

```python
torch.Tensor.requires_grad
```

### [paddle.Tensor.stop_gradient](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#stop-gradient)

```python
paddle.Tensor.stop_gradient
```

两者功能一致，且均无参数，Paddle 返回结果与 PyTorch 返回结果相反，具体如下：


### 转写示例
```python
# torch 写法
x.requires_grad

# paddle 写法
not x.stop_gradient
