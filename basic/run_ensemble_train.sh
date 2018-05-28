#!/usr/bin/env bash
inter_dir="inter_ensemble"

eargs=""
for num in 31 33 34 35 36 37 40 41 43 44 45 46; do
    eval_path="$inter_dir/eval-$num.json"
    eargs="$eargs $eval_path"
done
wait

python -m basic.ensemble_train $eargs
