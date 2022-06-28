# Paddle CI手册

## 整体介绍

当你提交一个PR`(Pull_Request)`，你的PR需要经过一些CI`(Continuous Integration)`，以触发`develop`分支的为例为你展示CI执行的顺序：

![ci_exec_order.png](../images/ci_exec_order.png)

如上图所示，提交一个`PR`，你需要：

- 签署CLA协议
- PR描述需要符合规范
- 通过不同平台`（Linux/Mac/Windows/XPU/NPU等）`的编译与单测
- 通过静态代码扫描工具的检测

**<font color=red>需要注意的是：如果你的PR只修改文档部分，你可以在commit中添加说明（commit message）以只触发文档相关的CI，写法如下：</font>**

```shell
# PR仅修改文档等内容，只触发PR-CI-Static-Check
git commit -m 'test=document_fix'
```

## 各流水线介绍

下面以触发`develop`分支为例，分平台对每条`CI`进行简单介绍。

### CLA

贡献者许可证协议[Contributor License Agreements](https://cla-assistant.io/PaddlePaddle/Paddle)是指当你要给Paddle贡献代码的时候，需要签署的一个协议。如果不签署那么你贡献给 Paddle 项目的修改，即`PR`会被 Github 标志为不可被接受，签署了之后，这个`PR`就是可以在 review 之后被接受了。

### CheckPRTemplate

检查PR描述信息是否按照模板填写。

- 通常10秒内检查完成，如遇长时间未更新状态，请re-edit一下PR描述重新触发该CI。

```markdown
### PR types
<!-- One of [ New features | Bug fixes | Function optimization | Performance optimization | Breaking changes | Others ] -->
(必填)从上述选项中，选择并填写PR类型
### PR changes
<!-- One of [ OPs | APIs | Docs | Others ] -->
(必填)从上述选项中，选择并填写PR所修改的内容
### Describe
<!-- Describe what this PR does -->
(必填)请填写PR的具体修改内容
```

### Linux平台

#### PR-CI-Clone

该CI主要是将当前PR的代码从GitHub clone到CI机器，方便后续的CI直接使用。

#### PR-CI-APPROVAL

该CI主要的功能是检测PR中的修改是否通过了审批。在其他CI通过之前，你可以无需过多关注该CI, 其他CI通过后会有相关人员进行review你的PR。

- 执行脚本：`paddle/scripts/paddle_build.sh assert_file_approvals`

#### PR-CI-Build

该CI主要是编译出当前PR的编译产物，并且将编译产物上传到BOS（百度智能云对象存储）中，方便后续的CI可以直接复用该编译产物。

- 执行脚本：`paddle/scripts/paddle_build.sh build_pr_dev`

#### PR-CI-Py3

该CI主要的功能是为了检测当前PR在CPU、Python3版本的编译与单测是否通过。

- 执行脚本：`paddle/scripts/paddle_build.sh cicheck_py37`

#### PR-CI-Coverage

该CI主要的功能是检测当前PR在GPU、Python3版本的编译与单测是否通过，同时增量代码需满足行覆盖率大于90%的要求。

- 编译脚本：`paddle/scripts/paddle_build.sh cpu_cicheck_coverage`
- 测试脚本：`paddle/scripts/paddle_build.sh gpu_cicheck_coverage`

#### PR-CE-Framework

该CI主要是为了测试P0级框架API与预测API的功能是否通过。此CI使用`PR-CI-Build`的编译产物，无需单独编译。

- 框架API测试脚本（[PaddlePaddle/PaddleTest](https://github.com/PaddlePaddle/PaddleTest)）：`PaddleTest/framework/api/run_paddle_ci.sh`
- 预测API测试脚本（[PaddlePaddle/PaddleTest](https://github.com/PaddlePaddle/PaddleTest)）：`PaddleTest/inference/python_api_test/parallel_run.sh `

#### PR-CI-ScienceTest

该CI主要是为了科学计算相关的单测是否通过。此CI使用`PR-CI-Build`的编译产物，无需单独编译。

- 测试脚本（[PaddlePaddle/PaddleScience](https://github.com/PaddlePaddle/PaddleScience)）：`PaddleScience/tests/test_examples/run.sh`

#### PR-CI-OP-benchmark

该CI主要的功能是PR中的修改是否会造成OP性能下降或者精度错误。此CI使用`PR-CI-Build`的编译产物，无需单独编译。

- 执行脚本：`tools/ci_op_benchmark.sh run_op_benchmark`

关于CI失败解决方案等详细信息可查阅[PR-CI-OP-benchmark Manual](https://github.com/PaddlePaddle/Paddle/wiki/PR-CI-OP-benchmark-Manual)

#### PR-CI-Model-benchmark

该CI主要的功能是检测PR中的修改是否会导致模型性能下降或者运行报错。此CI使用`PR-CI-Build`的编译产物，无需单独编译。

- 执行脚本：`tools/ci_model_benchmark.sh run_all`

关于CI失败解决方案等详细信息可查阅[PR-CI-Model-benchmark Manual](https://github.com/PaddlePaddle/Paddle/wiki/PR-CI-Model-benchmark-Manual)

#### PR-CI-Static-Check

该CI主要的功能是检查文档是否符合规范，检测`develop`分支与当前`PR`分支的增量的API英文文档是否符合规范，以及当变更API或OP时需要TPM approval。

- 编译脚本：`paddle/scripts/paddle_build.sh build_and_check_cpu`
- 示例文档检测脚本：`paddle/scripts/paddle_build.sh build_and_check_gpu`

#### PR-CI-Codestyle-Check

该CI主要的功能是检查提交代码是否符合规范，详细内容请参考[代码风格检查指南](./codestyle_check_guide_cn.html)。

- 执行脚本：`paddle/scripts/paddle_build.sh build_and_check_gpu`

#### PR-CI-infrt

该CI主要是为了检测infrt是否编译与单测通过

- 编译脚本：`paddle/scripts/infrt_build.sh build_only`
- 测试脚本：`paddle/scripts/infrt_build.sh test_only`

#### PR-CI-CINN

该CI主要是为了编译含CINN的Paddle，并运行Paddle-CINN对接的单测，保证训练框架进行CINN相关开发的正确性。

- 编译脚本：`paddle/scripts/paddle_build.sh build_only`
- 测试脚本：`paddle/scripts/paddle_build.sh test`

#### PR-CI-Inference

该CI主要的功能是为了检测当前PR对C++预测库与训练库的编译和单测是否通过。

- 编译脚本：`paddle/scripts/paddle_build.sh build_inference`
- 测试脚本：`paddle/scripts/paddle_build.sh gpu_inference`

#### PR-CI-GpuPS

该CI主要是为了保证GPUBOX相关代码合入后编译可以通过。

- 编译脚本：`paddle/scripts/paddle_build.sh build_gpubox`

### MAC

#### PR-CI-Mac-Python3

该CI是为了检测当前PR在MAC系统下python35版本的编译与单测是否通过，以及做develop与当前PR的单测增量检测，如有不同，提示需要approval。

- 执行脚本：`paddle/scripts/paddle_build.sh maccheck_py35`

### Windows

#### PR-CI-Windows

该CI是为了检测当前PR在Windows系统下MKL版本的GPU编译与单测是否通过，以及做develop与当前PR的单测增量检测，如有不同，提示需要approval。

- 执行脚本：`paddle/scripts/paddle_build.bat wincheck_mkl`

#### PR-CI-Windows-OPENBLAS

该CI是为了检测当前PR在Windows系统下OPENBLAS版本的CPU编译与单测是否通过。

- 执行脚本：`paddle/scripts/paddle_build.bat wincheck_openblas`

#### PR-CI-Windows-Inference

该CI是为了检测当前PR在Windows系统下预测模块的编译与单测是否通过。

- 执行脚本：`paddle/scripts/paddle_build.bat wincheck_inference`

### XPU机器

#### PR-CI-Kunlun

该CI主要的功能是检测PR中的修改能否在昆仑芯片上编译与单测通过。

- 执行脚本：`paddle/scripts/paddle_build.sh check_xpu_coverage`

### NPU机器

#### PR-CI-NPU

该CI主要是为了检测当前PR对NPU代码编译跟测试是否通过。

- 编译脚本：`paddle/scripts/paddle_build.sh build_only`
- 测试脚本：`paddle/scripts/paddle_build.sh gpu_cicheck_py35`

### Sugon-DCU机器

#### PR-CI-ROCM-Compile

该CI主要的功能是检测PR中的修改能否在曙光芯片上编译通过。

- 执行脚本：`paddle/scripts/musl_build/build_paddle.sh build_only`

### 静态代码扫描

#### PR-CI-iScan-C

该CI是为了检测当前PR的C++代码是否可以通过静态代码扫描。

#### PR-CI-iScan- Python

该CI是为了检测当前PR的Python代码是否可以通过静态代码扫描。



## CI失败如何处理
### CLA失败

- 如果你的cla一直是pending状态，那么需要等其他CI都通过后，点击 Close pull request ，再点击 Reopen pull request ，并等待几分钟（建立在你已经签署cla协议的前提下）；如果上述操作重复2次仍未生效，请重新提一个PR或评论区留言。
- 如果你的cla是失败状态，可能原因是你提交PR的账号并非你签署cla协议的账号，如下图所示：
![cla.png](./images/cla.png)
- 建议你在提交PR前设置：

```
git config –local user.email 你的邮箱
git config –local user.name 你的名字
```

### CheckPRTemplate失败

如果你的`CheckPRTemplate`状态一直未变化，这是由于通信原因状态未返回到GitHub。你只需要重新编辑一下PR描述保存后就可以重新触发该条CI，步骤如下：
![checkPRtemplate1.png](../images/checkPRtemplate1.png)
![checkPRTemplate2.png](../images/checkPRTemplate2.png)

### 其他CI失败

当你的`PR`的CI失败时，`paddle-bot`会在你的`PR`页面发出一条评论，同时此评论GitHub会同步到你的邮箱，让你第一时间感知到`PR`的状态变化（注意：只有第一条CI失败的时候会发邮件，之后失败的CI只会更新`PR`页面的评论。）

![paddle-bot-comment.png](../images/paddle-bot-comment.png)

![ci-details.png](../images/ci-details.png)

你可以通过点击`paddle-bot`评论中的CI名字，也可通过点击CI列表中的`Details`来查看CI的运行日志，如上图。通常运行日志的末尾会告诉你CI失败的原因。

由于网络代理、机器不稳定等原因，有时候CI的失败也并不是你的`PR`自身的原因，这时候你只需要rerun此CI即可（你需要将你的GitHub授权于效率云CI平台）。

![rerun.png](../images/rerun.png)

如果CI失败你无法判断原因，请联系 @[lelelelelez](https://github.com/lelelelelez)。

若遇到其他问题，请联系 @[lelelelelez](https://github.com/lelelelelez)。
