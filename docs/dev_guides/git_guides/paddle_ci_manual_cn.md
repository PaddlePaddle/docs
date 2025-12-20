# Paddle CI 测试详解

## 一、概述

持续集成（Continuous Integration，简称 CI）测试是项目开发与发布流水线中的重要一环。[PaddlePaddle/Paddle](https://github.com/PaddlePaddle/Paddle) 是一个多人协作开发项目，为了尽可能保证合入主干的代码质量，提高代码合入效率，开发者在提交一个 PR（Pull Request）时，将自动触发必要的 CI 测试任务，主要检测：

- 是否签署 CLA 协议。
- PR 描述是否符合规范。
- 是否通过不同平台（Linux/Mac/Windows/XPU/DCU 等）的编译与单测（单元测试）。
- 是否通过静态代码扫描工具的检测。

CI 测试包含的具体测试任务和执行顺序如下图所示：

![ci_exec_order.png](../images/ci_exec_order.png)

如上图所示，CI 测试任务将从左向右逐层执行，同一层任务并发执行。

提交 PR 后，请关注 PR 页面的 CI 测试进程，一般会在几个小时内完成。

- 测试项后出现绿色的对勾，表示本条测试项通过。
- 测试项后出现红色的叉号，并且后面显示 `Required`，则表示本条测试项不通过（不显示 `Required` 的任务未通过，也不影响代码合入，可不处理）。

> 注意 `Approval` 和 `Static-Check` 这两个 CI 测试项可能需要飞桨相关开发者 approve 才能通过，除此之外请确保其他每一项都通过，如果没有通过，请通过报错信息自查代码。

为了便于理解和处理 CI 测试问题，本文将逐条介绍各个 CI 测试项，并提供 CI 测试不通过的参考解决方法。

## 二、CI 测试项介绍

下面分平台对每条 CI 测试项进行简单介绍。

### **license/cla**

- **【条目描述】** 首次为 [PaddlePaddle/Paddle](https://github.com/PaddlePaddle/Paddle) 仓库贡献时，需要签署 [贡献者许可协议（Contributor License Agreement，CLA）](https://cla-assistant.io/PaddlePaddle/Paddle)，才可以合入代码。
- **【触发条件】** 自动触发。

### CheckPRTemplate

- **【条目描述】** 检查 PR 描述信息是否按照模板填写，模板如下：

```md
### PR Category
<!-- One of [ User Experience | Execute Infrastructure | Operator Mechanism | CINN | Custom Device | Performance Optimization | Distributed Strategy | Parameter Server | Communication Library | Auto Parallel | Inference | Environment Adaptation ] -->
（必填）从上述选项中，选择并填写 PR 分类
### PR Types
<!-- One of [ New features | Bug fixes | Improvements | Performance | BC Breaking | Deprecations | Docs | Devs | Not User Facing | Security | Others ] -->
（必填）从上述选项中，选择并填写 PR 类型
### Description
<!-- Describe what you’ve done -->
（必填）请填写 PR 的具体修改内容
```

- **【触发条件】** 自动触发。
- **【注意事项】** 通常 10 秒内检查完成，如遇长时间未更新状态，请编辑一下 PR 描述以重新触发。

### Paddle-Doc-Preview
- **【条目描述】** 构建文档并生成文档的预览。
- **【触发条件】** 自动触发。


### Linux 平台测试项

#### Auto-Parallel

- **【条目描述】** 检测飞桨自动并行 8 卡任务的功能、性能和显存。
- **【触发条件】** 当 PR 修改了 `tools/auto_parallel/target_path_lists.sh` 中的[命中路径](https://github.com/PaddlePaddle/Paddle/blob/develop/tools/auto_parallel/target_path_lists.sh)时触发。


#### Clone

- **【条目描述】** 将当前 PR 的代码从 GitHub Clone 到 CI 测试执行的机器，方便后续的 CI 直接使用。
- **【触发条件】** 自动触发。

#### Build

- **【条目描述】** 生成当前 PR 的编译产物，并将编译产物上传到 BOS（百度智能云对象存储）中，方便后续的 CI 可以直接复用该编译产物。
- **【执行脚本】** `paddle/scripts/paddle_build.sh build_pr_dev`
- **【触发条件】**
  - `Clone` 通过后自动触发。

#### CE-Framework

- **【条目描述】** 检测框架 API 与预测 API 的核心测试用例是否通过。
- **【执行脚本】**
  - [框架 API 测试](https://github.com/PaddlePaddle/PaddleTest)：`PaddleTest/framework/api/run_paddle_ci.sh`
  - [预测 API 测试](https://github.com/PaddlePaddle/PaddleTest)：`PaddleTest/inference/python_api_test/parallel_run.sh `
- **【触发条件】** `Build` 通过后自动触发，并且使用 `Build` 的编译产物，无需单独编译。

#### Model-benchmark

- **【条目描述】** 检测 PR 中的修改是否会导致模型性能下降或者运行报错。
- **【执行脚本】** `tools/ci_model_benchmark.sh run_all`
- **【触发条件】** `Build` 通过后自动触发，并且使用 `Build` 的编译产物，无需单独编译。
- **【注意事项】** 本条 CI 测试不通过的处理方法可查阅 [PR-CI-Model-benchmark Manual](https://github.com/PaddlePaddle/Paddle/wiki/PR-CI-Model-benchmark-Manual)。

#### Api-Benchmark

- **【条目描述】** 检测当前 PR 是否会造成 cpu 下动态图 api 调度性能下降。
- **【执行脚本】**
  - 拉取 PaddleTest 测试代码: git clone https://github.com/PaddlePaddle/PaddleTest.git
  - 进入执行目录: cd ./PaddleTest/framework/e2e/api_benchmark_new
  - 安装 CI 依赖: python -m pip install -r requirement.txt
  - 执行: python runner_user.py --yaml ./../yaml/ci_api_benchmark_fp32_cuda118_py310.yml
- **【触发条件】**
  - `Build` 通过后自动触发，并且使用 `Build` 的编译产物，无需单独编译
- **【补充说明】** CI 依赖./PaddleTest/framework/e2e/ci_api_benchmark_fp32_cuda118_py310.yml 中的配置文件运行。如果全部 case 运行耗时过长，需要单独调试某几个 api 的 case，可以把 ci_api_benchmark_fp32_cuda118_py310.yml 中，相关 api 的 yaml 配置信息，复制到你自己新建的.yml 配置文件中，使用 python runner_user.py --yaml your_yaml.yml 运行。性能测试需要对比你的 pr 合入前后，api 耗时增加的差异，也就是说需要运行两次进行对比调试。

#### Linux-CPU

- **【条目描述】** 检测当前 PR 在 CPU、Python3 版本的编译与单测是否通过。
- **【执行脚本】** `paddle/scripts/paddle_build.sh cicheck_py37`
- **【触发条件】** `Clone` 通过后自动触发。

#### Linux-XPU

- **【条目描述】** 检测 PR 中的修改能否在昆仑芯 XPU 上编译与单测通过。
- **【执行脚本】**
  - 构建 `paddle/scripts/paddle_build.sh check_xpu_coverage`
  - 测试 `paddle/scripts/paddle_build.sh test`
- **【触发条件】** `Clone` 通过后自动触发。

#### Linux-DCU

- **【条目描述】** 检测 PR 中的修改能否在海光 DCU 芯片上编译通过。
- **【执行脚本】** `paddle/scripts/musl_build/build_paddle.sh build_only`
- **【触发条件】** `Clone` 通过后自动触发。

#### Coverage

- **【条目描述】** 检测当前 PR 在 GPU、Python3 版本的编译与单测是否通过，同时增量代码需满足行覆盖率大于 90% 的要求。当 CI 完成后，`codecov-commeneter` 机器人会自动在 PR 下发表评论，展示当前 PR 的代码覆盖率情况，如下图所示：

![codecov-comment.png](../images/codecov-comment.png)

可点击跳转到 Codecov 网站查看更详细的覆盖率报告，如下图所示：

![codecov-report.png](../images/codecov-report.png)

- **【执行脚本】**
  - 编译脚本：`paddle/scripts/paddle_build.sh cpu_cicheck_coverage`
  - 测试脚本：`paddle/scripts/paddle_build.sh gpu_cicheck_coverage`
- **【触发条件】**
  - `Clone` 通过后自动触发。

#### Inference

- **【条目描述】** 检测当前 PR 对 C++ 预测库编译和单测是否通过。
- **【执行脚本】**
  - 编译脚本：`paddle/scripts/paddle_build.sh build_inference`
  - 测试脚本：`paddle/scripts/paddle_build.sh gpu_inference`
- **【触发条件】**
  - `Clone` 通过后自动触发。

#### Static-Check

- **【条目描述】** 检测`develop`分支与当前 `PR` 分支的增量 API 英文文档是否符合规范，以及当变更 API 或 OP 时检测是否经过了 TPM 审批（Approval）。
- **【执行脚本】**
  - 编译脚本：`paddle/scripts/paddle_build.sh build_and_check_cpu`
  - 测试脚本：`paddle/scripts/paddle_build.sh build_and_check_gpu`
- **【触发条件】** `Build` 通过后自动触发，并且使用 `Build` 的编译产物，无需单独编译。

#### Codestyle-Check

- **【条目描述】** 该 CI 主要的功能是检查提交代码是否符合规范，详细内容请参考[代码风格检查指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/git_guides/codestyle_check_guide_cn.html)。
- **【执行脚本】** `tools/codestyle/pre_commit.sh`
- **【触发条件】** `Clone` 通过后自动触发。
- **【注意事项】** 此 CI 需要检查代码风格，建议在提交 PR 之前安装 [pre-commit](https://pre-commit.com/)，可以在提交之前进行代码规范检查。

#### Approval

- **【条目描述】** 检测 PR 中的修改是否通过了审批（Approval）。
- **【执行脚本】** `paddle/scripts/paddle_build.sh assert_file_approvals`
- **【触发条件】** `Clone` 通过后自动触发。
- **【注意事项】** 在其他 CI 项通过前，无需过多关注该 CI，其他 CI 通过后飞桨相关开发者会进行审批。

#### PR-CI-SOT

- **【条目描述】** 检测当前 PR CPU、Python3.9-3.13 版本下的 SOT 单测是否通过。
- **【执行脚本】** `paddle/scripts/paddle_build.sh check_run_sot_ci`
- **【触发条件】**
  - `Clone` 通过后自动触发。
  - 必须修改下面路径中的文件才会触发

    ```bash
    paddle/fluid/operators/run_program_op.h
    paddle/fluid/operators/run_program_op.cu
    paddle/fluid/operators/run_program_op.cc
    paddle/fluid/eager/to_static
    paddle/fluid/pybind/
    python/
    test/sot
    ```

#### Distribute-stable

- **【条目描述】** 该 CI 主要的功能是检查提交代码是否引起编译错误、GPUBOX/LLM/自动并行架构的功能或精度错误
- **【执行脚本】** `paddle/scripts/paddle_build.sh distribute_test`
- **【触发条件】** `Clone` 通过后自动触发。
- **【注意事项】** 此 CI 的执行代码是在 PaddleNLP 的 repo 中的稳定版本分支 `<stable/paddle-ci>`，出现问题请优先在本地自测复现。


### macOS 平台测试项

#### Mac-CPU

- **【条目描述】** 检测当前 PR 在 MAC 系统下 Python 3.5 版本的编译与单测是否通过，并检测当前 PR 分支相比`develop`分支是否新增单测代码，如有不同，提示需要审批（Approval）。
- **【执行脚本】** `paddle/scripts/paddle_build.sh maccheck_py35`
- **【触发条件】** `Clone` 通过后自动触发。

### Windows 平台测试项

#### Windows

- **【条目描述】** 检测当前 PR 在 Windows GPU 环境下编译与单测是否通过，并检测当前 PR 分支相比`develop`分支是否新增单测代码，如有不同，提示需要审批（Approval）。
- **【执行脚本】** `paddle/scripts/paddle_build.bat wincheck_mkl`
- **【触发条件】**
  - 自动触发。
  - 当 `Windows-OPENBLAS` 任务失败时，会取消当前任务（因 `OPENBLAS` 失败，当前任务成功也无法进行代码合并，需要先排查 `OPENBLAS` 失败原因）。

#### Windows-OPENBLAS

- **【条目描述】** 检测当前 PR 在 Windows CPU 系统下编译与单测是否通过。
- **【执行脚本】** `paddle/scripts/paddle_build.bat wincheck_openblas`
- **【触发条件】** 自动触发。

#### Windows-Inference

- **【条目描述】** 检测当前 PR 在 Windows 系统下预测模块的编译与单测是否通过。
- **【执行脚本】** `paddle/scripts/paddle_build.bat wincheck_inference`
- **【触发条件】**
  - 自动触发。
  - 当 `Windows-OPENBLAS` 任务失败时，会取消当前任务（因 `OPENBLAS` 失败，当前任务成功也无法进行代码合并，需要先排查 `OPENBLAS` 失败原因）。

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

当 PR 中 CI 失败时，你可以通过点击 CI 列表标题来查看 CI 的运行日志，如下图所示。

![ci-details.png](../images/ci-details.png)

之后会跳转到日志查看页面，通常在运行日志的末尾会提示 CI 失败的原因，参考提示信息解决即可。可能的原因及处理办法如下：

#### (1) 网络原因

Paddle 编译、测试时需要下载一些第三方依赖，由于网络原因，可能会下载失败，导致编译、测试失败（如下图所示）。

![network_error.png](../images/network_error.png)

因此由于网络代理、机器不稳定等原因，遇到 Timeout、访问 503 等情况 ，可以尝试 rerun 失败的流水线即可。

对于有 write 权限的开发者，你可以直接点击流水线内的「`Re-run failed jobs`」按钮重新执行失败的 CI 任务，如下图所示：

![rerun.png](../images/rerun.png)

如果你没有 write 权限，可以在 PR 中回复一条 comment，内容为 `/re-run all-failed`，也可以达到重新执行失败 CI 任务的目的，如下图所示：

![rerun-via-comment.png](../images/rerun-via-comment.png)

> 注意：受限于 GitHub Actions 的机制，只有在同一 Workflow 中没有正在执行的任务时，才可以重新执行失败的任务。如果当前 Workflow 中有任务正在执行，则需要等待所有任务执行完成后，才能重新执行失败的任务。比如 `Linux-CPU` 和 `Mac-CPU` 同属 `CI` Workflow，如果 `Linux-CPU` 失败，而 `Mac-CPU` 还在执行中，则需要等待 `Mac-CPU` 执行完成后，才能重新执行 `Linux-CPU`。

#### (2) 合并代码失败

如果提交的代码较陈旧，可能会存在与其他 PR 修改同一文件同一行情况，存在冲突，导致 CI 无法进行 Merge develop，进而导致 CI 任务失败（如下图所示）。遇到该情况请本地执行 `git merge upstream develop` 再重新提交代码。

![merge_develop.png](../images/merge_develop.png)

#### (3) 取消任务

每条 CI 任务都设置有超时时间，如果任务失败并且页面显示灰色，应该是任务被取消。取消情况有三种:

1. 超时取消；
2. 当前 PR 有提交新的 commit，在运行或排队中的，旧 commit 任务全部取消；
3. 关联流水线取消，比如 `Linux-CPU` 任务失败会取消 `Linux-NPU` 等关联流水线。

解决办法：

第 1 种原因需要先排查是否是代码修改导致，确定不是代码原因导致的可以 rerun 此 CI 。

第 2 种无需要关心旧的 commit，新提交 commit 会继续执行 CI 任务，只需关心最新 commit 即可。

第 3 种需要排查 `Linux-CPU` 任务失败原因，确定不是代码原因导致的可以 rerun 所有失败 CI。

![cancel.png](../images/cancel.png)
