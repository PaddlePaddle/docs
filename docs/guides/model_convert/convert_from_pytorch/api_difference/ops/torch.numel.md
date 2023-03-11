## torch.numel
### [torch.numel](https://pytorch.org/docs/stable/generated/torch.numel.html?highlight=numel#torch.numel)

```python
torch.numel(input)
```

### [paddle.Tensor.size](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#size)

```python
paddle.Tensor.size
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | -            | 输入的 Tensor。                   |


### 转写示例
```python
# Pytorch 写法
torch.numel(a)

# Paddle 写法
a.size
```
