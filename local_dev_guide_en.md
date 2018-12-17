# Guide of local development

You will learn how to develop programs in local under the guide of this document.

## Requirements of code
- Please refer to the coding note format of [Doxygen](http://www.stack.nl/~dimitri/doxygen/) 
- Make sure that option of builder `WITH_STYLE_CHECK` is on and the build could pass through the code style check.
- Unit test is a need for all codes.
- Pass through all unit tests.
- Please follow [regulations of submitting code](#regulations of submitting code).

It tells you how to submit code as follows.
## [Fork](https://help.github.com/articles/fork-a-repo/)

Transfer to the homepage of Github [PaddlePaddle](https://github.com/PaddlePaddle/Paddle),and then click button`Fork` to generate the repository under your own file directory,such as <https://github.com/USERNAME/Paddle>.

## Clone

Clone remote repository to local:

```bash
➜  git clone https://github.com/USERNAME/Paddle
➜  cd Paddle
```


## Create local branch

At present [Git stream branch model](http://nvie.com/posts/a-successful-git-branching-model/) is applied to Paddle to undergo task of development,test,release and maintenance.Please refer to [branch regulation of Paddle](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/releasing_process.md#paddle-分支规范) about details。

All development tasks of feature and bug fix should be finished in a new branch which is extended from `develop` branch.

Create and change into a new branch with command `git checkout -b`.

```bash
➜  git checkout -b my-cool-stuff
```

Atentions should be paid that current branch be clean before checkout,otherwise the untracked files will also be shifted to the new branch,which could be viewed with command `git status`.

## Use `pre-commit` hook

Paddle developers manage Git pre-commit hook with [pre-commit](http://pre-commit.com/),which could help us to fomulate source code(C++，Python) and automatically review certain details,such as only one EOL of every file and no large file added to Git,before your commit.

`pre-commit` test is part of unit tests of Travis-CI.PR that doesn't meet the requirement of hook can't be commit to Paddle.Install `pre-commit` first and then run it on current directory：

```bash
➜  pip install pre-commit
➜  pre-commit install
```

You can modify the format of C/C++ source code with `clang-format` in Paddle.Make sure the version of `clang-format` is above 3.8.

Note：There are differences between the installation of `yapf` with `pip install pre-commit` and that with `conda install -c conda-forge pre-commit`.Paddle developers use `pip install pre-commit`.

## Starting the development

I delete one line of README.md and create a new file at the example.

Check current status with `git status`,followed by the appearance of changes under current directory.Meanwhile,you can also check the specific modifications of the file with `git diff`.

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

## Constructure and test

It needs a various of development tools to bulid Paddle Paddle source code and generate documents.For convenience,our standard development procedure is to put these tools together into a Docker image,called *development mirror*,usually named as `paddle:latest-dev` or `paddle:[version tag]-dev`,such as `paddle:0.11.0-dev`.Then all that need `cmake && make` ,such as IDE configuration,are replaced by `docker run paddle:latest-dev`.

You need to bulid this development mirror under the root directory of source code directory tree 

```bash
➜  docker build -t paddle:latest-dev .
```

Then you can start building PaddlePaddle source code with this development mirror.For example,to build a PaddlePaddle which is not dependent on GPU but in support of AVX commands and including unit test,you can:

```bash
➜  docker run -v $(pwd):/paddle -e "WITH_GPU=OFF" -e "WITH_AVX=ON" -e "WITH_TESTING=ON" paddle:latest-dev
```

If you want to build PaddlePaddle based on Python3,you can:

```bash
➜  docker run -v $(pwd):/paddle -e "PY_VERSION=3.5" -e "WITH_FLUID_ONLY=ON" -e "WITH_GPU=OFF" -e "WITH_AVX=ON" -e "WITH_TESTING=ON" paddle:latest-dev
```

Except for the build of PaddlePaddle as `./build/libpaddle.so` and the output of `./build/paddle.deb` file, there is a output of `build/Dockerfile`.What we need to do is to package the PaddlePaddle as a *produce mirror*（`paddle:prod`）with following command.

```bash
➜  docker build -t paddle:prod -f build/Dockerfile .
```

Run all unit tests with following command:

```bash
➜  docker run -it -v $(pwd):/paddle paddle:latest-dev bash -c "cd /paddle/build && ctest"
```

Please refer to [Installation and run with Docker](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/v2/build_and_install/docker_install_cn.rst) about more information of constructure and test.

## Commit

Next we cancel the modification of README.md,and submit new added test file.

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

Commit message is a must at evry submit to Git so that other members would learn about the modification that you have made,which could be finished with commang `git commit`.

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

<b> <font color="red">Attention needs to be paid：you need to add commit message to touch CI test with following commands.</font> </b>

```bash
# Touch CI single test of develop branch
➜  git commit -m "test=develop"

# Touch CI single test of release/1.1 branch
➜  git commit -m "test=release/1.1"
```

## Keep the latest local repository

It needs to keep up with the latest code of original repository (<https://github.com/PaddlePaddle/Paddle>）before Pull Request.

Check the name of current remote repository with `git remote`.

```bash
➜  git remote
origin
➜  git remote -v
origin	https://github.com/USERNAME/Paddle (fetch)
origin	https://github.com/USERNAME/Paddle (push)
```

origin is the name of remote repository that we clone,which is also the Paddle under your own account.Next we create a remote host of an original Paddle and name it upstream.

```bash
➜  git remote add upstream https://github.com/PaddlePaddle/Paddle
➜  git remote
origin
upstream
```

Get the latest code of upstream and update current branch.

```bash
➜  git fetch upstream
➜  git pull upstream develop
```

## Push to remote repository

Submit local modification to GitHub (https://github.com/USERNAME/Paddle).

```bash
# submit it to the branch my-cool-stuff of remote repository origin
➜  git push origin my-cool-stuff
```