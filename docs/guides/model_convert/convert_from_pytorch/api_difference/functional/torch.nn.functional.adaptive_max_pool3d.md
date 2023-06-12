## [ 参数不一致 ]torch.nn.functional.adaptive_max_pool3d

### [torch.nn.functional.adaptive_max_pool3d](https://pytorch.org/docs/stable/generated/torch.nn.functional.adaptive_max_pool3d.html?highlight=adaptive_max_pool3d#torch.nn.functional.adaptive_max_pool3d)

```python
torch.nn.functional.adaptive_max_pool3d(input,
                                        output_size,
                                        return_indices=False)
```

### [paddle.nn.functional.adaptive_max_pool3d](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/adaptive_max_pool3d_cn.html)

```python
paddle.nn.functional.adaptive_max_pool3d(x,
                                        output_size,
                                        return_mask=False,
                                        name=None)
```

两者功能一致，当返回索引时数据类型不一致， 具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input           | x           |  表示输入的 Tensor ，仅参数名不一致。               |
| output_size           |  output_size           | 表示输出 Tensor 的大小，仅参数名不一致。               |
| return_indices           |  return_mask          | 表示是否返回最大值的索引，仅参数名不一致。               |
| return           |  return          | 表示返回值，当返回索引时，PyTorch 返回值类型为 int64，Paddle 返回值类型为 int32，需要进行转写。             |

### 转写示例
#### return：返回索引
```python
# Pytorch 写法
result, indices = nn.functional.adaptive_max_pool3d(x, 1, True)

# Paddle 写法
result, indices = paddle.nn.functional.adaptive_max_pool3d(x, 1, return_mask=True)
indices = indices.astype('int64')
```
