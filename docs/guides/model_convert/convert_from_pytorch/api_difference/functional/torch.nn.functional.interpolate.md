## torch.nn.functional.interpolate

### [torch.nn.functional.interpolate](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/interpolate_cn.html#interpolate)

```python
torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False)
```

### [paddle.nn.functional.interpolate](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/interpolate_cn.html#interpolate)

```python
paddle.nn.functional.interpolate(x, size=None, scale_factor=None, mode='nearest', align_corners=False, align_mode=0, data_format='NCHW', name=None)
```

两者功能一致，torch 参数多，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 表示输入的 Tensor                                       |
| recompute_scale_factor       | -         | 是否重计算 scale_factor                      |
| antialias     | -            | 是否使用反别名                                           |
| -             | align_mode   | 双线性插值的索引计算方法                                   |
| -             | data_format  | 数据的格式                                              |

### 转写示例

```python
# torch 写法
torch.nn.functional.interpolate(input, size=(100, 100), scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False)

# paddle 写法
paddle.nn.functional.interpolate(x, size=(100, 100), scale_factor=None, mode='nearest', align_corners=False, align_mode=0, data_format='NCHW', name=None)
```
