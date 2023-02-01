## torch.permute
### [torch.permute](https://pytorch.org/docs/stable/generated/torch.permute.html?highlight=permute#torch.permute)

```python
torch.permute(input, 
                dims)
```

### [paddle.transpose](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/transpose_cn.html#transpose)

```python
paddle.transpose(x, 
                perm, 
                name=None)
```
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input        | x            | 输入的多维 Tensor。                   |
| dims         | perm         | dims 长度必须和 input 的维度相同，并依照 dims 中数据进行重排。 |
