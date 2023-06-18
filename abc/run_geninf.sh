#!/bin/bash
#run example:
#  bash run_geninf.sh  cmake-build-debug/abc  /home/liujunfeng/benchmarks/comb-exp  10m

#circuits=("C880") # "C6288" "C5315" "C7552" "64b_mult" "aes" "square" "sqrt")
# 循环遍历参数列表

files=$(find "$2" -name "*.aig")
for filepath in ${files[@]}
do
  filename=$(basename "$filepath")
  name="${filename%.*}"
  echo "process $name"
  command="read_lib -v /home/liujunfeng/SLAP/ML-Mapper/abc/asap7_clean.lib; read /home/liujunfeng/benchmarks/comb-exp/${name}.aig; gen_inf -E ../data/test/${name}_cell_emb.csv -N ../data/test/${name}_node_emb.csv -C ../data/test/${name}_cut_emb.csv"
  outputs=$(timeout $3 $1 -c "$command";)
  echo $outputs
done

files=$(find "$2" -name "*.blif")
for filepath in ${files[@]}
do
  filename=$(basename "$filepath")
  name="${filename%.*}"
  echo "process $name"
  command="read_lib -v /home/liujunfeng/SLAP/ML-Mapper/abc/asap7_clean.lib; read /home/liujunfeng/benchmarks/comb-exp/${name}.blif; gen_inf -E ../data/test/${name}_cell_emb.csv -N ../data/test/${name}_node_emb.csv -C ../data/test/${name}_cut_emb.csv"
  outputs=$(timeout $3 $1 -c "$command";)
  echo $outputs
done