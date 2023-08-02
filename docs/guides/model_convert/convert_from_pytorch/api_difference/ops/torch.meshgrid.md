## [torch 参数更多]torch.meshgrid
### [torch.meshgrid](https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid)

```python
torch.meshgrid(*tensors, indexing=None)
```

### [paddle.meshgrid](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/meshgrid_cn.html#meshgrid)

```python
paddle.meshgrid(*args, **kargs)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| tensors       | args         | 输入的 Tensor 列表，仅参数名不一致。                                      |
| indexing      | -            | tensor 的组合模式。Paddle 无此参数，需要转写。                                        |

### 转写示例
#### indexing：tensor 的组合模式
```python
# Pytorch 写法 (indexing 为‘ij’时)
torch.meshgrid(x, y, indexing='ij')

# Paddle 写法
paddle.meshgrid(x, y)

# Pytorch 写法 (indexing 为‘xy’时)
torch.meshgrid(x, y, indexing='xy')

# Paddle 写法
list([i.T for i in paddle.meshgrid(x, y)])
```
