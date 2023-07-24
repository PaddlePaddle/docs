## [torch 参数更多]torch.asarray

### [torch.asarray](https://pytorch.org/docs/stable/generated/torch.asarray.html#torch.asarray)

```python
torch.asarray(obj, *, dtype=None, device=None, copy=None, requires_grad=False)
```

### [paddle.to_tensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/to_tensor_cn.html)

```python
paddle.to_tensor(data, dtype=None, place=None, stop_gradient=True)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle  | 备注                                                                     |
| ------------- | ------------- | ------------------------------------------------------------------------ |
| obj           | data          | 初始化 Tensor 的数据，仅参数名不一致。                                   |
| dtype         | dtype         | 创建 Tensor 的数据类型。                                                 |
| device        | place         | 创建 Tensor 的设备位置。                                                 |
| copy          | -             | 是否和原 Tensor 共享内存，Paddle 无此参数，暂无转写方式。                |
| requires_grad | stop_gradient | 是否阻断 Autograd 的梯度传导，PyTorch 和 Paddle 取值相反，需要进行转写。 |

### 转写示例

#### requires_grad 参数：是否阻断 Autograd 的梯度传导

```python
# PyTorch 写法:
torch.asarray(x, requires_grad=False)

# Paddle 写法:
paddle.to_tensor(x, stop_gradient=True)
```
