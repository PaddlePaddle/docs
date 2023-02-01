## torch.unique_consecutive
### [torch.unique_consecutive](https://pytorch.org/docs/stable/generated/torch.unique_consecutive.html?highlight=unique_consecutive#torch.unique_consecutive)

```python
torch.unique_consecutive(input, 
                        return_inverse=False, 
                        return_counts=False, 
                        dim=None)
```

### [paddle.unique_consecutive](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/unique_consecutive_cn.html#unique-consecutive)

```python
paddle.unique_consecutive(x, 
                            return_inverse=False, 
                            return_counts=False, 
                            axis=None, 
                            dtype='int64', 
                            name=None)
```
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input        | x            | 输入的 Tensor。                   |
| dim          | axis         | PyTorch表示是否不阻断梯度传导，PaddlePaddle表示是否阻断梯度传导。 |