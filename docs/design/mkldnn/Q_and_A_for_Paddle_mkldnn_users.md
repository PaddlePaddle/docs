Please help enrich this doc, to reduce similar questions

# Intel Xeon generations

- What is SKX
- What is CLX
- What ICX
- What is CPX
- What is SPR
https://ark.intel.com/content/www/us/en/ark/products/codename/189143/products-formerly-cooper-lake.html?wapkw=cooper%20lake

# Paddle mkldnn(oneDNN) related issues

- Why elementwise_add is sometimes very slow
    Do the profiling and check the log if the elementwise_add on the top and see what is top time consuming.
    Check in the profiling log if there is some ops that does not have mkldnn kernel, if there is, maybe this op is targeting op that need further oneDNN support.
    Especially if there is interpolate op in the model, Since there is slight difference in computation method of oneDNN interpolate and Paddle interpolate, so interpolate by default will always call Paddle native interpolate. You need to add config->PassBuilder()->AppendPadd("interpolate_mkldnn_pass");

