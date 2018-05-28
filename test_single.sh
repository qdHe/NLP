#!/usr/bin/env bash
python -m basic.cli --len_opt --cluster
wait
python squad/evaluate-v1.1.py data/squad/dev-v1.1.json out/basic/00/answer/test-016000.json
