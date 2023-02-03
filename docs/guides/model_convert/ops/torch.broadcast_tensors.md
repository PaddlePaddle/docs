## torch.broadcast_tensors
### [torch.broadcast_tensors](https://pytorch.org/docs/stable/generated/torch.broadcast_tensors.html?highlight=broadcast_tensors#torch.broadcast_tensors)

```python
torch.broadcast_tensors(*tensors)
```

### [paddle.broadcast_tensors](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/broadcast_tensors_cn.html#broadcast-tensors)

```python
paddle.broadcast_tensors(inputs,
                        name=None)
```
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| *tensors      | inputs       | 一组输入 Tensor。                   |


### 功能差异

#### 使用方式
***PyTorch***：tensors 为可变参数。
***PaddlePaddle***：inputs 为 list(Tensor)或 tuple(Tensor)的形式。
