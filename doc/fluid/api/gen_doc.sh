#!/bin/bash

# fluid
for module in layers dataset clip metrics executor initializer io nets optimizer profiler regularizer transpiler backward profiler unique_name dygraph
do
  python gen_doc.py --module_name ${module} --module_prefix ${module} --output ${module} --output_name fluid --to_multiple_files True
  python gen_module_index.py ${module}  fluid.${module}
done

python gen_doc.py --module_name "" --module_prefix "" --output fluid --output_name fluid --to_multiple_files True
python gen_module_index.py fluid  fluid


# index.rst
python gen_index.py

