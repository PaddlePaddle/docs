## [ torch 参数更多 ]torch.Tensor.log_normal_

### [torch.Tensor.log_normal_](https://pytorch.org/docs/stable/generated/torch.Tensor.log_normal_.html#torch-tensor-log-normal)

```python
torch.Tensor.log_normal_(mean=1, std=2, *, generator=None)
```

### [paddle.Tensor.log_normal_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#log_normal-mean-1-0-std-2-0-shape-none-name-none)

```python
paddle.Tensor.log_normal_(mean=1.0, std=2.0, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                                |
| ------------- | ------------ | ----------------------------------------------------------------------------------- |
| generator     | -            | 用于采样的伪随机数生成器， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。    |

### 转写示例

```python
# torch 写法
x = torch.randn(2, 3)
y = x.log_normal_()

# paddle 写法
x = paddle.randn([2, 3])
y = x.log_normal_()
```
