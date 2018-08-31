# Introduction
Fluiddoc consolidates all the documentations related to Paddle. It supplies the contents to PaddlePaddle.org via CI. 

# Architecture
FluidDoc submodules Paddle, Book, Models, Mobile and Anakin under `external` folder. All submodules should be put under `external` as standard practice. 

Fluiddoc then uses them as references to load up the documents. The FluidDoc constructs the whole doc-tree under the `FluidDoc/doc/fluid` folder. The entry point is `FluidDoc/doc/fluid/index_cn.rst` and `FluidDoc/doc/fluid/index_en.rst`

When a release branch is pushed to Github, Travis-CI will start automatically to compile documents and deploy documents to the server. 

## Note: 
FluidDoc needs Paddle python module to compile API documents. Unfortunately, compiling Paddle python module takes longer time Travis CI permits. Usually Travis CI will fail due because of timeout. That's why there three jobs on Travis, two of them are to build libraries. Once the libraries are cached on the Travis, next build will be a lot faster.

# Publish New release
1. Checkout a new release branch. The branch name should follow `release/<version>`
1. Update the documentations on the submodules or within FluidDoc
1. Update the submodules under `external` folder and commit the changes.
1. Git push the branch to Github, Travis CI will start several builds to publish the documents

