## torch.Tensor.backward
### [torch.Tensor.backward](https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html?highlight=backward#torch.Tensor.backward)

```python
torch.Tensor.backward(gradient=None, retain_graph=None, create_graph=False, inputs=None)
```

### [paddle.Tensor.backward](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#backward-grad-tensor-none-retain-graph-false)

```python
paddle.Tensor.backward(grad_tensor=None, retain_graph=False)
```

两者功能类似，torch 参数多，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| gradient      | grad_tensor  | 梯度初始值                                               |
| retain_graph  | retain_graph | 是否保留计算图                                           |
| create_graph  | -            | 创建导数的计算图，用于计算更高阶的导数，没有该功能，无法转写                       |
| inputs        | -            | 需要计算梯度的 Tensor 列表，没有该功能，无法转写                                  |
