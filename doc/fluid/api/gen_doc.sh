#!/bin/bash

for module in layers dataset clip metrics executor initializer io nets optimizer profiler regularizer transpiler backward profiler unique_name dygraph
do
  python gen_doc.py --module_name ${module} --module_prefix ${module} --output ${module} --output_name fluid --to_multiple_files True
  python gen_module_index.py ${module}  fluid.${module}
done

python gen_doc.py --module_name "" --module_prefix "" --output fluid --output_name fluid --to_multiple_files True
python gen_module_index.py fluid  fluid

for module in math random stat
do
  python gen_doc.py --module_name ${module} --module_prefix ${module} --output ${module} --output_name tensor --to_multiple_files True --output_dir tensor
  python gen_module_index.py tensor.${module} ${module}
done

python gen_module_index.py tensor paddle.tensor

for module in loss
do
  python gen_doc.py --module_name ${module} --module_prefix ${module} --output ${module} --output_name nn --to_multiple_files True --output_dir nn
  python gen_module_index.py nn.${module} ${module}
done

python gen_module_index.py nn paddle.nn

python gen_index.py

