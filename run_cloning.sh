#!/bin/bash

python src/behavior_cloning.py -f cloning_100 -t CloseMicrowave
python src/behavior_cloning.py -f cloning_100 -t PushButton
python src/behavior_cloning.py -f cloning_100 -t UnplugCharger
python src/behavior_cloning.py -f cloning_100 -t TakeLidOffSaucepan
python src/behavior_cloning.py -f cloning_10 -t CloseMicrowave &
python src/behavior_cloning.py -f cloning_10 -t PushButton &
python src/behavior_cloning.py -f cloning_10 -t UnplugCharger &
python src/behavior_cloning.py -f cloning_10 -t TakeLidOffSaucepan &
wait
python src/evaluate.py -f cloning_10 -t CloseMicrowave &
python src/evaluate.py -f cloning_100 -t CloseMicrowave &
python src/evaluate.py -f cloning_10 -t PushButton &
python src/evaluate.py -f cloning_100 -t PushButton &
wait
python src/evaluate.py -f cloning_10 -t UnplugCharger &
python src/evaluate.py -f cloning_100 -t UnplugCharger &
python src/evaluate.py -f cloning_10 -t TakeLidOffSaucepan &
python src/evaluate.py -f cloning_100 -t TakeLidOffSaucepan &
