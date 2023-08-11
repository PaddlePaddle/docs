## [ torch 参数更多 ] torch.Tensor.copy_

### [torch.Tensor.copy_](https://pytorch.org/docs/stable/generated/torch.Tensor.copy_.html#torch.Tensor.copy_)

```python
torch.Tensor.copy_(src, non_blocking=False)
```

### [paddle.assign](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/assign_cn.html#assign)

```python
paddle.assign(x, output=None)
```

两者功能类似，torch 参数多，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| src           | x            | 待复制的 tensor，仅参数名不一致。                                         |
| non_blocking  | -            | 用于控制 cpu 和 gpu 数据的异步复制。Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。       |
| -             | output       | 输出 Tensor，Pytorch 无此参数，Paddle 需将其设置为调用 copy_类方法的 Tensor 。        |


### 转写示例

```python
# torch 写法
y.copy_(x)

# paddle 写法
paddle.assign(x, output=y)
```
