## [torch 参数更多 ]torch.unique
### [torch.unique](https://pytorch.org/docs/stable/generated/torch.unique.html?highlight=unique#torch.unique)

```python
torch.unique(input,
             sorted,
             return_inverse,
             return_counts,
             dim=None)
```

### [paddle.unique](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/unique_cn.html#unique)

```python
paddle.unique(x,
              return_index=False,
              return_inverse=False,
              return_counts=False,
              axis=None,
              dtype='int64',
              name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 表示输入 Tensor。  |
| sorted        | -            | 表示是否按升序返回排列，PaddlePaddle 无此参数。暂无转写方式  |
| -             | return_index | 表示是否返回独有元素在输入 Tensor 中的索引，PyTorch 无此参数。Paddle 保持默认即可。  |
| return_inverse| return_inverse| 表示是否返回输入 Tensor 的元素对应在独有元素中的索引。  |
| return_counts | return_counts| 表示是否返回每个独有元素在输入 Tensor 中的个数。  |
| dim           | axis         | 表示指定选取独有元素的轴。  |
| -         | dtype            | 表示返回值的类型，PyTorch 无此参数， Paddle 保持默认即可。  |
