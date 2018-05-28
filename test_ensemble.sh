#!/usr/bin/env bash
source_path="data/squad/dev-v1.1.json"
target_path="ensemble.json"
inter_dir="inter_ensemble"
root_dir="save"

parg=""
marg=""
if [ "$3" = "debug" ]
then
    parg="-d"
    marg="--debug"
fi


eargs=""
for num in 0 1 2 3 4 5 6 7 8 9 10 11 12; do
    load_path="$root_dir/$num/save"
    shared_path="$root_dir/$num/shared.json"
    eval_path="$inter_dir/eval-$num.json"
    eargs="$eargs $eval_path"
done
wait

# Ensemble
python3 -m basic.ensemble --data_path $inter_dir/data_single.json --shared_path $inter_dir/shared_single.json -o $target_path $eargs

python squad/evaluate-v1.1.py $source_path $target_path
