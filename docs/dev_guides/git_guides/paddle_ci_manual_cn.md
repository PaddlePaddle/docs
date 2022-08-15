# Paddle CI 测试详解

## 一、概述

持续集成（Continuous Integration，简称 CI）测试是项目开发与发布流水线中的重要一环。[PaddlePaddle/Paddle](https://github.com/PaddlePaddle/Paddle) 是一个多人协作开发项目，为了尽可能保证合入主干的代码质量，提高代码合入效率，开发者在提交一个 PR（Pull Request）时，将自动触发必要的 CI 测试任务，主要检测：

- 是否签署 CLA 协议。
- PR 描述是否符合规范。
- 是否通过不同平台`（Linux/Mac/Windows/XPU/NPU/DCU 等）`的编译与单测（单元测试）。
- 是否通过静态代码扫描工具的检测。

CI 测试包含的具体测试任务和执行顺序如下图所示：

![ci_exec_order.png](../images/ci_exec_order.png)

如上图所示，CI 测试任务将从左向右逐层执行，同一层任务并发执行。

> 说明：如果 PR 中仅修改了文档内容，可在 `git commit` 时在描述信息中添加 `'test=document_fix'`关键字，如 `git commit -m 'message, test=document_fix',`即可只触发 PR-CI-Static-Check，仅检查文档是否符合规范，不做其他代码检查。

提交 PR 后，请关注 PR 页面的 CI 测试进程，一般会在几个小时内完成。

- 测试项后出现绿色的对勾，表示本条测试项通过。
- 测试项后出现红色的叉号，并且后面显示 `Required`，则表示本条测试项不通过（不显示 `Required` 的任务未通过，也不影响代码合入，可不处理）。

为了便于理解和处理 CI 测试问题，本文将逐条介绍各个 CI 测试项，并提供 CI 测试不通过的参考解决方法。

## 二、CI 测试项介绍

下面分平台对每条 CI 测试项进行简单介绍。

### **license/cla**

- **【条目描述】**首次为 [PaddlePaddle/Paddle](https://github.com/PaddlePaddle/Paddle) 仓库贡献时，需要签署 [贡献者许可协议（Contributor License Agreement，CLA）](https://cla-assistant.io/PaddlePaddle/Paddle)，才可以合入代码。
- **【触发条件】**自动触发。

### CheckPRTemplate

- **【条目描述】**检查 PR 描述信息是否按照模板填写，模板如下：

```md
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

- **【触发条件】**自动触发。
- **【注意事项】**通常 10 秒内检查完成，如遇长时间未更新状态，请编辑一下 PR 描述以重新触发。

### Linux 平台测试项

#### PR-CI-Clone

- **【条目描述】**将当前 PR 的代码从 GitHub Clone 到 CI 测试执行的机器，方便后续的 CI 直接使用。
- **【触发条件】**自动触发。

#### PR-CI-Build

- **【条目描述】**生成当前 PR 的编译产物，并将编译产物上传到 BOS（百度智能云对象存储）中，方便后续的 CI 可以直接复用该编译产物。
- **【执行脚本】**`paddle/scripts/paddle_build.sh build_pr_dev`
- **【触发条件】**
  - `PR-CI-Clone`通过后自动触发。
  - 当 PR-CI-Py3 任务失败时，会取消当前任务（因 PR-CI-Py3 失败，当前任务成功也无法进行代码合并，需要先排查 PR-CI-Py3 失败原因）。

#### PR-CE-Framework

- **【条目描述】**检测框架 API 与预测 API 的核心测试用例是否通过。
- **【执行脚本】**
  - [框架 API 测试](https://github.com/PaddlePaddle/PaddleTest)：`PaddleTest/framework/api/run_paddle_ci.sh`
  - [预测 API 测试](https://github.com/PaddlePaddle/PaddleTest)：`PaddleTest/inference/python_api_test/parallel_run.sh `
- **【触发条件】**`PR-CI-Build`通过后自动触发，并且使用`PR-CI-Build`的编译产物，无需单独编译。

#### PR-CI-Model-benchmark

- **【条目描述】**检测 PR 中的修改是否会导致模型性能下降或者运行报错。
- **【执行脚本】**`tools/ci_model_benchmark.sh run_all`
- **【触发条件】**`PR-CI-Build`通过后自动触发，并且使用`PR-CI-Build`的编译产物，无需单独编译。
- **【注意事项】**本条 CI 测试不通过的处理方法可查阅 [PR-CI-Model-benchmark Manual](https://github.com/PaddlePaddle/Paddle/wiki/PR-CI-Model-benchmark-Manual)。

#### PR-CI-OP-benchmark

- **【条目描述】**检测 PR 中的修改是否会造成 OP 性能下降或者精度错误。
- **【执行脚本】**`tools/ci_op_benchmark.sh run_op_benchmark`
- **【触发条件】**`PR-CI-Build`通过后自动触发，并且使用`PR-CI-Build`的编译产物，无需单独编译。
- **【注意事项】**本条 CI 测试不通过的处理方法可查阅 [PR-CI-OP-benchmark Manual](https://github.com/PaddlePaddle/Paddle/wiki/PR-CI-OP-benchmark-Manual)。

#### PR-CI-Py3

- **【条目描述】**检测当前 PR 在 CPU、Python3 版本的编译与单测是否通过。
- **【执行脚本】**`paddle/scripts/paddle_build.sh cicheck_py37`
- **【触发条件】**`PR-CI-Clone`通过后自动触发。

#### PR-CI-Coverage

- **【条目描述】**检测当前 PR 在 GPU、Python3 版本的编译与单测是否通过，同时增量代码需满足行覆盖率大于 90% 的要求。可在 PR 页面点击该 CI 后的 details 查看覆盖率，如下图所示：


- **【条目描述】**检测当前 PR 的 C++ 代码是否通过 [静态代码扫描](https://clang-analyzer.llvm.org/)。
- **【触发条件】**自动触发。

#### PR-CI-iScan-Python

- **【条目描述】**检测当前 PR 的 Python 代码是否通过 [静态代码扫描](https://pylint.pycqa.org/)。
- **【触发条件】**自动触发。

## 三、CI 失败如何处理

### 3.1 CLA 失败

- 如果 PR 中 license/cla 检测项一直是 pending 状态，那么需要等其他 CI 项都通过后，点击 `Close pull request`，再点击 `Reopen pull request`，并等待几分钟（前提是你已经签署 CLA 协议）。如果上述操作重复 2 次仍未生效，请重新提一个 PR 或在评论区留言。
- 如果 PR 中 license/cla 是失败状态，可能原因是提交 PR 的 GitHub 账号与签署 CLA 协议的账号不一致，如下图所示：

![cla.png](../images/cla.png)

建议在提交 PR 前设置：

```plain
git config --local user.email 你的 GitHub 邮箱
git config --local user.name 你的 GitHub 名字
```

### 3.2 CheckPRTemplate 失败

如果 PR 中`CheckPRTemplate`状态一直未变化，这是由于通信原因，状态未返回到 GitHub。只需要重新编辑保存一下 PR 描述后，就可以重新触发该条 CI，步骤如下：

![checkPRtemplate1.png](../images/checkPRtemplate1.png)

![checkPRTemplate2.png](../images/checkPRTemplate2.png)

### 3.3 其他 CI 失败

当 PR 中 CI 失败时，`paddle-bot`会在 PR 页面发出一条评论，同时 GitHub 会发送到你的邮箱，让你第一时间感知到 PR 的状态变化。

> 注意：只有 PR 中第一条 CI 失败的时候会发邮件，之后失败的 CI 项只会更新在 PR 页面的评论中。

可通过点击`paddle-bot`评论中的 CI 名称，也可通过点击 CI 列表中的`Details`来查看 CI 的运行日志，如下图所示。

![paddle-bot-comment.png](../images/paddle-bot-comment.png)

![ci-details.png](../images/ci-details.png)

之后会跳转到日志查看页面，通常在运行日志的末尾会提示 CI 失败的原因，参考提示信息解决即可。由于网络代理、机器不稳定等原因，有时候 CI 的失败也并不是 PR 自身的原因，此时只需要`重新构建`此 CI 即可（需要将你的 GitHub 授权于效率云 CI 平台），如下图所示。

![rerun.png](../images/rerun.png)
