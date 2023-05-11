## [torch 参数更多] torch.hub.list

### [torch.hub.list](https://pytorch.org/docs/1.13/hub.html?highlight=hub+list#torch.hub.list)

```python
torch.hub.list(github,
                force_reload=False,
                skip_validation=False,
                trust_repo=None)
```

### [paddle.hub.list](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/hub/list_cn.html)

```python
paddle.hub.list(repo_dir,
                source='github',
                force_reload=False)
```

两者功能一致但参数不一致，部分参数名不同，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| github        | repo_dir      |repo 地址，支持 git 地址形式和 local 地址，参数名不同。|
| force_reload   | force_reload |指定是否强制拉取，默认值: False。             |
| skip_validation| -          |检查由 github 参数指定的分支或提交是否属于存储库所有者,默认为 False，Paddle 无此参数，无需转写。|
| trust_repo    | -            |在 v1.14 中被移除，Paddle 无此参数，可直接删除。|
|-              |source        |指定 repo 托管的位置，支持 github、gitee 和 local，默认值：github。|
