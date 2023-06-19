#!/bin/bash
#run example:
#  bash run_ourmapping.sh  cmake-build-debug/abc  10m



timestamp=$(date +%Y%m%d%H%M%S)
csv="our_res_${timestamp}.csv"
touch "$csv"

echo "name, area(delay), delay(delay), area(QoR), delay(QoR)" >> $csv

# handle aig
#circuits_aig=("s444_comb" "C6288" "s526_comb" "s9234_1_comb"  "adder"  "C880"  "sin"   "aes"  "C7552"  "max"  "sqrt" "multiplier"  "bar" "s5378_comb" "C5315")
circuits_aig=("cavlc"  "ctrl"  "dec"  "i2c"  "int2float"  "mem_ctrl" "priority"  "router")
for name in "${circuits_aig[@]}"
do
    echo "process $name"
    # delay oriented
    command="read_lib -v /home/liujunfeng/SLAP/ML-Mapper/abc/asap7_clean.lib; read /home/liujunfeng/benchmarks/control-exp/${name}.aig;  map -r -P ../data/test/${name}_predictions_2.txt; topo; stime;"
    outputs=$(timeout $2 $1 -c "$command";)
    numbers=($(echo $outputs | grep -Eo '[0-9]+(\.[0-9]+)?'))
    size=${#numbers[@]}

    # QoR oriented
    command2="read_lib -v /home/liujunfeng/SLAP/ML-Mapper/abc/asap7_clean.lib; read /home/liujunfeng/benchmarks/control-exp/${name}.aig;  map -P ../data/test/${name}_predictions_2.txt; topo; stime;"
    outputs2=$(timeout $2 $1 -c "$command2";)
    numbers2=($(echo $outputs2 | grep -Eo '[0-9]+(\.[0-9]+)?'))
    size2=${#numbers2[@]}
    ret="$name, ${numbers[$size-8]},  ${numbers[$size-3]}, ${numbers2[$size2-8]},  ${numbers2[$size2-3]}"
    echo $ret >> $csv
done


#circuits_blif=("mul64-booth"  "mul32-booth" "rc64b"  "rc256b")
#circuits_blif=("64b_mult")
#for name in "${circuits_blif[@]}"
#do
#    echo "process $name"
#    # delay oriented
#    command="read_lib -v /home/liujunfeng/SLAP/ML-Mapper/abc/asap7_clean.lib; read /home/liujunfeng/benchmarks/comb-exp/${name}.blif;  map -r -P ../data/test/${name}_predictions_2.txt; topo; stime;"
#    outputs=$(timeout $2 $1 -c "$command";)
#    numbers=($(echo $outputs | grep -Eo '[0-9]+(\.[0-9]+)?'))
#    size=${#numbers[@]}
#
#    # QoR oriented
#    command2="read_lib -v /home/liujunfeng/SLAP/ML-Mapper/abc/asap7_clean.lib; read /home/liujunfeng/benchmarks/comb-exp/${name}.blif;  map -P ../data/test/${name}_predictions_2.txt; topo; stime;"
#    outputs2=$(timeout $2 $1 -c "$command2";)
#    numbers2=($(echo $outputs2 | grep -Eo '[0-9]+(\.[0-9]+)?'))
#    size2=${#numbers2[@]}
#    ret="$name, ${numbers[$size-8]},  ${numbers[$size-3]}, ${numbers2[$size2-8]},  ${numbers2[$size2-3]}"
#    echo $ret >> $csv
#done