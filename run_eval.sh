#!/bin/bash

python src/evaluate.py -f dagger -t CloseMicrowave &
python src/evaluate.py -f iwr -t CloseMicrowave &
python src/evaluate.py -f ceiling_full -t CloseMicrowave &
wait
python src/evaluate.py -f dagger -t PushButton &
python src/evaluate.py -f iwr -t PushButton &
python src/evaluate.py -f ceiling_full -t PushButton &
