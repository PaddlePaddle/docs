## torch.empty_like
### [torch.empty_like](https://pytorch.org/docs/stable/generated/torch.empty_like.html?highlight=empty_like#torch.empty_like)

```python
torch.empty_like(input,
                 *,
                 dtype=None,
                 layout=None,
                 device=None,
                 requires_grad=False,
                 memory_format=torch.preserve_format)
```

### [paddle.empty_like](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/tensor/creation/empty_like_cn.html#empty-like)

```python
paddle.empty_like(x,
                  dtype=None,
                  name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 表示输入 Tensor。                                     |
| layout        | -            | 表示布局方式，PaddlePaddle 无此参数，一般对网络训练结果影响不大，可直接删除。                   |
| device        | -            | 表示 Tensor 存放位置，PaddlePaddle 无此参数，一般对网络训练结果影响不大，可直接删除。                   |
| requires_grad | -            | 表示是否不阻断梯度传导，PaddlePaddle 无此参数。 |
| pin_memeory   | -            | 表示是否使用锁页内存，PaddlePaddle 无此参数。           |


### 转写示例
#### requires_grad：是否需要求反向梯度，需要修改该 Tensor 的 stop_gradient 属性
```python
# Pytorch 写法
x = torch.empty_like((2, 3), requires_grad=True)

# Paddle 写法
x = paddle.empty_like([2, 3])
x.stop_gradient = False
```

#### pin_memory：是否分配到固定内存上
```python
# Pytorch 写法
x = torch.empty_like((2, 3), pin_memory=True)

# Paddle 写法
x = paddle.empty_like([2, 3]).pin_memory()
```
