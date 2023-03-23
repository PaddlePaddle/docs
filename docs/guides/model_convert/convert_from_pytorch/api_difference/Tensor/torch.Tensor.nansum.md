## [ 一致的参数 ] torch.Tensor.nansum
### [torch.Tensor.nansum](https://pytorch.org/docs/1.13/generated/torch.Tensor.nansum.html?highlight=nansum#torch.Tensor.nansum)

```python
Tensor.nansum(dim=None, keepdim=False, dtype=None)
#示例代码
import torch

x = torch.Tensor([[1, float('-nan'), 1], [2, float('-nan'), 2]])
out1 = x.nansum()
print(out1)    # 6
out2 = x.nansum(dim=-1)
print(out2)    #[2,4]
out3 = x.nansum(dim=-2)
print(out3)    #[3,0,3]
```

### [paddle.Tensor.nansum](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html)未找到文档

```python
paddle.Tensor.nansum(axis=None, keepdim=False, dtype=None)
#示例代码
import paddle

x = paddle.to_tensor([[1, float('-nan'), 1], [2, float('-nan'), 2]]).astype("float64")
out1 = x.nansum()
print(out1)    #[6]
out2 = x.nansum(axis=-1)
print(out2)    #[2,4]
out3 = x.nansum(axis=-2)
print(out3)    #[3,0,3]
```
两者功能一致，返回张量中元素的和，其中nan值记为0

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| dim          | axis         | 需要求和的维度                                     |
| keepdim          | keepdim         | 结果是否需要保持维度                                     |
| dtype          | dtype         | 返回的类型                                     |

