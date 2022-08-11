# Paddle CI 手册

## 整体介绍

当你提交一个 PR`(Pull_Request)`，你的 PR 需要经过一些 CI`(Continuous Integration)`，以触发`develop`分支的为例为你展示 CI 执行的顺序：

![ci_exec_order.png](../images/ci_exec_order.png)

如上图所示，提交一个`PR`，你需要：

- 签署 CLA 协议
- PR 描述需要符合规范
- 通过不同平台`（Linux/Mac/Windows/XPU/NPU 等）`的编译与单测
- 通过静态代码扫描工具的检测

**<font color=red>需要注意的是：如果你的 PR 只修改文档部分，你可以在 commit 中添加说明（commit message）以只触发文档相关的 CI，写法如下：</font>**

```shell
# PR 仅修改文档等内容，只触发 PR-CI-Static-Check
git commit -m 'test=document_fix'
```

## 各流水线介绍

下面以触发`develop`分支为例，分平台对每条`CI`进行简单介绍。

### CLA

贡献者许可证协议[Contributor License Agreements](https://cla-assistant.io/PaddlePaddle/Paddle)是指当你要给 Paddle 贡献代码的时候，需要签署的一个协议。如果不签署那么你贡献给 Paddle 项目的修改，即`PR`会被 GitHub 标志为不可被接受，签署了之后，这个`PR`就是可以在 review 之后被接受了。

### CheckPRTemplate

检查 PR 描述信息是否按照模板填写。

- 通常 10 秒内检查完成，如遇长时间未更新状态，请 re-edit 一下 PR 描述重新触发该 CI。

```markdown
### PR types
<!-- One of [ New features | Bug fixes | Function optimization | Performance optimization | Breaking changes | Others ] -->
(必填)从上述选项中，选择并填写 PR 类型
### PR changes
<!-- One of [ OPs | APIs | Docs | Others ] -->
(必填)从上述选项中，选择并填写 PR 所修改的内容
### Describe
<!-- Describe what this PR does -->
(必填)请填写 PR 的具体修改内容
```

### Linux 平台

#### PR-CI-Clone

该 CI 主要是将当前 PR 的代码从 GitHub clone 到 CI 机器，方便后续的 CI 直接使用。

#### PR-CI-APPROVAL

该 CI 主要的功能是检测 PR 中的修改是否通过了审批。在其他 CI 通过之前，你可以无需过多关注该 CI, 其他 CI 通过后会有相关人员进行 review 你的 PR。

- 执行脚本：`paddle/scripts/paddle_build.sh assert_file_approvals`

#### PR-CI-Build

该 CI 主要是编译出当前 PR 的编译产物，并且将编译产物上传到 BOS（百度智能云对象存储）中，方便后续的 CI 可以直接复用该编译产物。

- 执行脚本：`paddle/scripts/paddle_build.sh build_pr_dev`

#### PR-CI-Py3

该 CI 主要的功能是为了检测当前 PR 在 CPU、Python3 版本的编译与单测是否通过。

- 执行脚本：`paddle/scripts/paddle_build.sh cicheck_py37`

#### PR-CI-Coverage

该 CI 主要的功能是检测当前 PR 在 GPU、Python3 版本的编译与单测是否通过，同时增量代码需满足行覆盖率大于 90%的要求。

- 编译脚本：`paddle/scripts/paddle_build.sh cpu_cicheck_coverage`
- 测试脚本：`paddle/scripts/paddle_build.sh gpu_cicheck_coverage`

#### PR-CE-Framework

该 CI 主要是为了测试 P0 级框架 API 与预测 API 的功能是否通过。此 CI 使用`PR-CI-Build`的编译产物，无需单独编译。

- 框架 API 测试脚本（[PaddlePaddle/PaddleTest](https://github.com/PaddlePaddle/PaddleTest)）：`PaddleTest/framework/api/run_paddle_ci.sh`
- 预测 API 测试脚本（[PaddlePaddle/PaddleTest](https://github.com/PaddlePaddle/PaddleTest)）：`PaddleTest/inference/python_api_test/parallel_run.sh `

#### PR-CI-OP-benchmark

该 CI 主要的功能是 PR 中的修改是否会造成 OP 性能下降或者精度错误。此 CI 使用`PR-CI-Build`的编译产物，无需单独编译。

- 执行脚本：`tools/ci_op_benchmark.sh run_op_benchmark`

关于 CI 失败解决方案等详细信息可查阅[PR-CI-OP-benchmark Manual](https://github.com/PaddlePaddle/Paddle/wiki/PR-CI-OP-benchmark-Manual)

#### PR-CI-Model-benchmark

该 CI 主要的功能是检测 PR 中的修改是否会导致模型性能下降或者运行报错。此 CI 使用`PR-CI-Build`的编译产物，无需单独编译。

- 执行脚本：`tools/ci_model_benchmark.sh run_all`

关于 CI 失败解决方案等详细信息可查阅[PR-CI-Model-benchmark Manual](https://github.com/PaddlePaddle/Paddle/wiki/PR-CI-Model-benchmark-Manual)

#### PR-CI-Static-Check

该 CI 主要的功能是检查文档是否符合规范，检测`develop`分支与当前`PR`分支的增量的 API 英文文档是否符合规范，以及当变更 API 或 OP 时需要 TPM approval。

- 编译脚本：`paddle/scripts/paddle_build.sh build_and_check_cpu`
- 示例文档检测脚本：`paddle/scripts/paddle_build.sh build_and_check_gpu`

#### PR-CI-Codestyle-Check

该 CI 主要的功能是检查提交代码是否符合规范，详细内容请参考[代码风格检查指南](./codestyle_check_guide_cn.html)。

- 执行脚本：`paddle/scripts/paddle_build.sh build_and_check_gpu`

#### PR-CI-CINN

该 CI 主要是为了编译含 CINN 的 Paddle，并运行 Paddle-CINN 对接的单测，保证训练框架进行 CINN 相关开发的正确性。

- 编译脚本：`paddle/scripts/paddle_build.sh build_only`
- 测试脚本：`paddle/scripts/paddle_build.sh test`

#### PR-CI-Inference

该 CI 主要的功能是为了检测当前 PR 对 C++预测库与训练库的编译和单测是否通过。

- 编译脚本：`paddle/scripts/paddle_build.sh build_inference`
- 测试脚本：`paddle/scripts/paddle_build.sh gpu_inference`

#### PR-CI-GpuPS

该 CI 主要是为了保证 GPUBOX 相关代码合入后编译可以通过。

- 编译脚本：`paddle/scripts/paddle_build.sh build_gpubox`

### MAC

#### PR-CI-Mac-Python3

该 CI 是为了检测当前 PR 在 MAC 系统下 python35 版本的编译与单测是否通过，以及做 develop 与当前 PR 的单测增量检测，如有不同，提示需要 approval。

- 执行脚本：`paddle/scripts/paddle_build.sh maccheck_py35`

### Windows

#### PR-CI-Windows

该 CI 是为了检测当前 PR 在 Windows 系统下 MKL 版本的 GPU 编译与单测是否通过，以及做 develop 与当前 PR 的单测增量检测，如有不同，提示需要 approval。

- 执行脚本：`paddle/scripts/paddle_build.bat wincheck_mkl`

#### PR-CI-Windows-OPENBLAS

该 CI 是为了检测当前 PR 在 Windows 系统下 OPENBLAS 版本的 CPU 编译与单测是否通过。

- 执行脚本：`paddle/scripts/paddle_build.bat wincheck_openblas`

#### PR-CI-Windows-Inference

该 CI 是为了检测当前 PR 在 Windows 系统下预测模块的编译与单测是否通过。

- 执行脚本：`paddle/scripts/paddle_build.bat wincheck_inference`

### XPU 机器

#### PR-CI-Kunlun

该 CI 主要的功能是检测 PR 中的修改能否在昆仑芯片上编译与单测通过。

- 执行脚本：`paddle/scripts/paddle_build.sh check_xpu_coverage`

### NPU 机器

#### PR-CI-NPU

该 CI 主要是为了检测当前 PR 对 NPU 代码编译跟测试是否通过。

- 编译脚本：`paddle/scripts/paddle_build.sh build_only`
- 测试脚本：`paddle/scripts/paddle_build.sh gpu_cicheck_py35`

### Sugon-DCU 机器

#### PR-CI-ROCM-Compile

该 CI 主要的功能是检测 PR 中的修改能否在曙光芯片上编译通过。

- 执行脚本：`paddle/scripts/musl_build/build_paddle.sh build_only`

### 静态代码扫描

#### PR-CI-iScan-C

该 CI 是为了检测当前 PR 的 C++代码是否可以通过静态代码扫描。

#### PR-CI-iScan- Python

该 CI 是为了检测当前 PR 的 Python 代码是否可以通过静态代码扫描。



## CI 失败如何处理
### CLA 失败

- 如果你的 cla 一直是 pending 状态，那么需要等其他 CI 都通过后，点击 Close pull request ，再点击 Reopen pull request ，并等待几分钟（建立在你已经签署 cla 协议的前提下）；如果上述操作重复 2 次仍未生效，请重新提一个 PR 或评论区留言。
- 如果你的 cla 是失败状态，可能原因是你提交 PR 的账号并非你签署 cla 协议的账号，如下图所示：
![cla.png](./images/cla.png)
- 建议你在提交 PR 前设置：

```
git config –local user.email 你的邮箱
git config –local user.name 你的名字
```

### CheckPRTemplate 失败

如果你的`CheckPRTemplate`状态一直未变化，这是由于通信原因状态未返回到 GitHub。你只需要重新编辑一下 PR 描述保存后就可以重新触发该条 CI，步骤如下：
![checkPRtemplate1.png](../images/checkPRtemplate1.png)
![checkPRTemplate2.png](../images/checkPRTemplate2.png)

### 其他 CI 失败

当你的`PR`的 CI 失败时，`paddle-bot`会在你的`PR`页面发出一条评论，同时此评论 GitHub 会同步到你的邮箱，让你第一时间感知到`PR`的状态变化（注意：只有第一条 CI 失败的时候会发邮件，之后失败的 CI 只会更新`PR`页面的评论。）

![paddle-bot-comment.png](../images/paddle-bot-comment.png)

![ci-details.png](../images/ci-details.png)

你可以通过点击`paddle-bot`评论中的 CI 名字，也可通过点击 CI 列表中的`Details`来查看 CI 的运行日志，如上图。通常运行日志的末尾会告诉你 CI 失败的原因。

由于网络代理、机器不稳定等原因，有时候 CI 的失败也并不是你的`PR`自身的原因，这时候你只需要 rerun 此 CI 即可（你需要将你的 GitHub 授权于效率云 CI 平台）。

![rerun.png](../images/rerun.png)
