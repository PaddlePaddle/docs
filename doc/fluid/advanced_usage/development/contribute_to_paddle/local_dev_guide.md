# 本地开发指南

本文将指导您如何在本地进行代码开发

## 代码要求
- 代码注释请遵守 [Doxygen](http://www.stack.nl/~dimitri/doxygen/) 的样式。
- 确保编译器选项 `WITH_STYLE_CHECK` 已打开，并且编译能通过代码样式检查。
- 所有代码必须具有单元测试。
- 通过所有单元测试。
- 请遵守[提交代码的一些约定](#提交代码的一些约定)。

以下教程将指导您提交代码。
## [Fork](https://help.github.com/articles/fork-a-repo/)

跳转到[PaddlePaddle](https://github.com/PaddlePaddle/Paddle) GitHub首页，然后单击 `Fork` 按钮，生成自己目录下的仓库，比如 <https://github.com/USERNAME/Paddle>。

## 克隆（Clone）

将远程仓库 clone 到本地：

```bash
➜  git clone https://github.com/USERNAME/Paddle
➜  cd Paddle
```


## 创建本地分支

Paddle 目前使用[Git流分支模型](http://nvie.com/posts/a-successful-git-branching-model/)进行开发，测试，发行和维护，具体请参考 [Paddle 分支规范](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/releasing_process.md#paddle-分支规范)。

所有的 feature 和 bug fix 的开发工作都应该在一个新的分支上完成，一般从 `develop` 分支上创建新分支。

使用 `git checkout -b` 创建并切换到新分支。

```bash
➜  git checkout -b my-cool-stuff
```

值得注意的是，在 checkout 之前，需要保持当前分支目录 clean，否则会把 untracked 的文件也带到新分支上，这可以通过 `git status` 查看。

## 使用 `pre-commit` 钩子

Paddle 开发人员使用 [pre-commit](http://pre-commit.com/) 工具来管理 Git 预提交钩子。 它可以帮助我们格式化源代码（C++，Python），在提交（commit）前自动检查一些基本事宜（如每个文件只有一个 EOL，Git 中不要添加大文件等）。

`pre-commit`测试是 Travis-CI 中单元测试的一部分，不满足钩子的 PR 不能被提交到 Paddle，首先安装并在当前目录运行它：

```bash
➜  pip install pre-commit
➜  pre-commit install
```

Paddle 使用 `clang-format` 来调整 C/C++ 源代码格式，请确保 `clang-format` 版本在 3.8 以上。

注：通过`pip install pre-commit`和`conda install -c conda-forge pre-commit`安装的`yapf`稍有不同的，Paddle 开发人员使用的是`pip install pre-commit`。

## 开始开发

在本例中，我删除了 README.md 中的一行，并创建了一个新文件。

通过 `git status` 查看当前状态，这会提示当前目录的一些变化，同时也可以通过 `git diff` 查看文件具体被修改的内容。

```bash
➜  git status
On branch test
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   README.md

Untracked files:
  (use "git add <file>..." to include in what will be committed)

	test

no changes added to commit (use "git add" and/or "git commit -a")
```

## 构建和测试

编译 PaddlePaddle 的源码以及生成文档需要多种开发工具。为了方便大家，我们的标准开发流程是把这些工具都装进一个Docker image，称为*开发镜像*，通常名字是 `paddle:latest-dev` 或者 `paddle:[version tag]-dev` 如 `paddle:0.11.0-dev`。然后所有用 `cmake && make` 的地方（比如IDE配置里）都用 `docker run paddle:latest-dev`来代替。

如要build这个开发镜像，在源码目录树的根目录中运行：

```bash
➜  docker build -t paddle:latest-dev .
```

随后可以用这个开发镜像开始build PaddlePaddle的源码。比如如果要build一个不依赖GPU，但是支持AVX指令集，并且包括unit tests的PaddlePaddle，可以：

```bash
➜  docker run -v $(pwd):/paddle -e "WITH_GPU=OFF" -e "WITH_AVX=ON" -e "WITH_TESTING=ON" paddle:latest-dev
```

如果你需要在此基础上编译基于Python3的PaddlePaddle，可以：

```bash
➜  docker run -v $(pwd):/paddle -e "PY_VERSION=3.5" -e "WITH_FLUID_ONLY=ON" -e "WITH_GPU=OFF" -e "WITH_AVX=ON" -e "WITH_TESTING=ON" paddle:latest-dev
```

这个过程除了编译PaddlePaddle为 `./build/libpaddle.so`，并且输出一个 `./build/paddle.deb`文件之外，还会输出一个 `build/Dockerfile`。我们只需要运行下面命令把编译好的PaddlePaddle打包成一个*生产镜像*（`paddle:prod`）：

```bash
➜  docker build -t paddle:prod -f build/Dockerfile .
```

如果要运行所有的单元测试，可以用如下命令：

```bash
➜  docker run -it -v $(pwd):/paddle paddle:latest-dev bash -c "cd /paddle/build && ctest"
```

关于构建和测试的更多信息，请参见[使用Docker安装运行](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/v2/build_and_install/docker_install_cn.rst)。

## 提交（commit）

接下来我们取消对 README.md 文件的改变，然后提交新添加的 test 文件。

```bash
➜  git checkout -- README.md
➜  git status
On branch test
Untracked files:
  (use "git add <file>..." to include in what will be committed)

	test

nothing added to commit but untracked files present (use "git add" to track)
➜  git add test
```

Git 每次提交代码，都需要写提交说明，这可以让其他人知道这次提交做了哪些改变，这可以通过`git commit` 完成。

```bash
➜  git commit
CRLF end-lines remover...............................(no files to check)Skipped
yapf.................................................(no files to check)Skipped
Check for added large files..............................................Passed
Check for merge conflicts................................................Passed
Check for broken symlinks................................................Passed
Detect Private Key...................................(no files to check)Skipped
Fix End of Files.....................................(no files to check)Skipped
clang-formater.......................................(no files to check)Skipped
[my-cool-stuff c703c041] add test file
 1 file changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 233
```

<b> <font color="red">需要注意的是：您需要在commit中添加说明（commit message）以触发CI单测，写法如下：</font> </b>

```bash
# 触发develop分支的CI单测
➜  git commit -m "test=develop"

# 触发release/1.1分支的CI单侧
➜  git commit -m "test=release/1.1"
```

## 保持本地仓库最新

在准备发起 Pull Request 之前，需要同步原仓库（<https://github.com/PaddlePaddle/Paddle>）最新的代码。

首先通过 `git remote` 查看当前远程仓库的名字。

```bash
➜  git remote
origin
➜  git remote -v
origin	https://github.com/USERNAME/Paddle (fetch)
origin	https://github.com/USERNAME/Paddle (push)
```

这里 origin 是我们 clone 的远程仓库的名字，也就是自己用户名下的 Paddle，接下来我们创建一个原始 Paddle 仓库的远程主机，命名为 upstream。

```bash
➜  git remote add upstream https://github.com/PaddlePaddle/Paddle
➜  git remote
origin
upstream
```

获取 upstream 的最新代码并更新当前分支。

```bash
➜  git fetch upstream
➜  git pull upstream develop
```

## Push 到远程仓库

将本地的修改推送到 GitHub 上，也就是 https://github.com/USERNAME/Paddle。

```bash
# 推送到远程仓库 origin 的 my-cool-stuff 分支上
➜  git push origin my-cool-stuff
```
