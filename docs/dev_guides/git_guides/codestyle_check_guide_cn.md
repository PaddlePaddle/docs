# 代码风格检查指南

整洁、规范的代码风格，能够保证代码的可读性、易用性和健壮性。Paddle 使用 [pre-commit](http://pre-commit.com/) 工具进行代码风格检查。它可以帮助检查提交代码的不规范问题并格式化（当前会检查 C++、Python 和 CMake 语言的代码）。诸如 ruff、cpplint 等工具能提前发现代码的潜在静态逻辑错误，提高开发效率。

在 Paddle CI 中，由 PR-CI-Codestyle-Check 流水线对提交的 PR 进行代码风格检查，若该流水线执行失败，PR 将**无法合入**到 Paddle 仓库。此时需要根据流水线日志的报错信息，在本地修改代码，再次提交。一般情况下，本地使用 `pre-commit` 进行代码风格检查的结果和 PR-CI-Codestyle-Check 流水线结果是一致的。下面介绍 `pre-commit` 的本地安装与使用方法。

Paddle 目前使用的 pre-commit 版本是 2.17.0。首先安装并在当前目录运行它：

```bash
➜  pip install pre-commit==2.17.0
➜  pre-commit install
```

> 注：通过 `pip install pre-commit` 和 `conda install -c conda-forge pre-commit` 安装的 `pre-commit` 稍有不同，Paddle 开发人员使用的是 `pip install pre-commit`。

在使用 `git commit` 提交修改时，pre-commit 将自动检查修改文件的代码规范，并对不符合规范的文件进行格式化。此时，`git commit` 并未执行成功，需要将 pre-commit 对文件的修改添加到暂存区，再次 commit，直到 pre-commit 代码检查通过后，本次提交才算完成。
例如，对 `Paddle/paddle/phi/kernels/abs_kernel.h` 修改后，提交 commit，通过 `git diff` 查看，会发现 clang-format 修改了该文件，需要添加修改后，再次 `git commit`，完成本次提交。

```bash
➜  git diff
diff --git a/paddle/phi/kernels/abs_kernel.h b/paddle/phi/kernels/abs_kernel.h
index e79216a021..7e06204845 100644
--- a/paddle/phi/kernels/abs_kernel.h
+++ b/paddle/phi/kernels/abs_kernel.h
@@ -22,4 +22,5 @@ namespace phi {
 template <typename T, typename Context>
 void AbsKernel(const Context& ctx, const DenseTensor& x, DenseTensor* out);

-}  // namespace phi
+void test_func();
+}  // namespace
➜  git add paddle/phi/kernels/abs_kernel.h
➜  git commit -m "test"
check for added large files..............................................Passed
check for merge conflicts................................................Passed
check for broken symlinks............................(no files to check)Skipped
detect private key.......................................................Passed
fix end of files.........................................................Passed
sort simple yaml files...............................(no files to check)Skipped
trim trailing whitespace.................................................Passed
CRLF end-lines remover...................................................Passed
Tabs remover (C++).......................................................Passed
Tabs remover (Python)................................(no files to check)Skipped
copyright_checker........................................................Passed
black................................................(no files to check)Skipped
ruff.................................................(no files to check)Skipped
clang-format.............................................................Failed
- hook id: clang-format
- files were modified by this hook
cpplint..................................................................Passed
clang-tidy...............................................................Passed
auto-generate-cmakelists.............................(no files to check)Skipped
cmake-format.........................................(no files to check)Skipped
CMake Lint...........................................(no files to check)Skipped
➜  git diff
diff --git a/paddle/phi/kernels/abs_kernel.h b/paddle/phi/kernels/abs_kernel.h
index 7e06204845..c1b803b44f 100644
--- a/paddle/phi/kernels/abs_kernel.h
+++ b/paddle/phi/kernels/abs_kernel.h
@@ -23,4 +23,4 @@ template <typename T, typename Context>
 void AbsKernel(const Context& ctx, const DenseTensor& x, DenseTensor* out);

 void test_func();
-}  // namespace
+}  // namespace phi
➜  git add paddle/phi/kernels/abs_kernel.h
➜  git commit -m "test"
➜  git log
commit xxx
Author: xxx
Date:   xxx

    test

...
```

目前 pre-commit 主要执行 C++、Python、CMake 语言的代码规范和格式化，以及 git 相关的通用检查和格式化。所有的检查工具信息如下：

| 检查工具名称 | 作用 | 当前版本 |
|---|---|---|
| [pre-commit](https://github.com/pre-commit/pre-commit) | hook 管理工具 | 2.17.0 |
| [pre-commit/pre-commit-hooks](https://github.com/pre-commit/pre-commit-hooks) | pre-commit 官方支持的 hook，执行一些通用检查 | 4.4.0 |
| [Lucas-C/pre-commit-hooks](https://github.com/Lucas-C/pre-commit-hooks.git) | 社区维护的一些通用的 hook，含将 CRLF 改为 LF、移除 Tab 等 hook | 1.5.1 |
| [copyright_checker](https://github.com/PaddlePaddle/Paddle/blob/develop/tools/codestyle/copyright.hook) | Copyright 检查 | 本地脚本 |
| [black](https://github.com/psf/black) | Python 代码格式化 | 24.8.0 |
| [ruff](https://github.com/astral-sh/ruff) | Python 代码风格检查 | 0.6.1 |
| [clang-format](https://github.com/llvm/llvm-project/tree/main/clang/tools/clang-format) | C++ 代码格式化 | 13.0.0 |
| [cpplint](https://github.com/cpplint/cpplint) | C++ 代码风格检查 | 1.6.0 |
| [clang-tidy](https://github.com/llvm/llvm-project/tree/main/clang-tools-extra/clang-tidy) | C++ 代码风格检查 | 15.0.2.1 |
| [cmake-format](https://github.com/cheshirekow/cmake-format-precommit) | CMake 代码格式化 | 0.6.13 |
| [cmake-lint](https://github.com/PFCCLab/cmake-lint-paddle)| CMake 代码风格检查 | 1.5.1 |

> 注：这些工具可能会更新，详细配置请查看：[https://github.com/PaddlePaddle/Paddle/blob/develop/.pre-commit-config.yaml](https://github.com/PaddlePaddle/Paddle/blob/develop/.pre-commit-config.yaml)。

## FAQ
1. pre-commit==2.17.0 要求 Python>=3.6.1，建议使用较高版本的 Python。
2. 在首次 commit 时，pre-commit 需要初始化环境，执行时间会稍长一些，大概在 3min 左右。
3. 在首次 commit 前，请先升级 pip，并使用 pypi 官方镜像源，否则，可能会导致 clang-format 或者 cmake-lint 安装失败。命令如下：
```bash
➜  pip install --upgrade pip
➜  pip config set global.index-url https://pypi.python.org/simple
```
