## 1. Tournament -> uniform -> Mutation -> Shc
#python GA.py --dataset karate --num_label 2 --pop_size 100 --generation 50 --use_local True --selection_method ts --crossover_method uniform --local_method shc --fig_size 20 --run 10
#python GA.py --dataset dolphins --num_label 5 --pop_size 400 --generation 100 --use_local True --selection_method ts --crossover_method uniform --local_method shc --fig_size 20 --run 10
#python GA.py --dataset book --num_label 3 --pop_size 400 --generation 100 --use_local True --selection_method ts --crossover_method uniform --local_method shc --fig_size 20 --run 10
#python GA.py --dataset football --num_label 12 --pop_size 500 --generation 100 --use_local True --selection_method ts --crossover_method uniform --local_method shc --fig_size 20 --run 10
#python GA.py --dataset jazz --num_label 4 --pop_size 400  --generation 200 --use_local True --selection_method ts --crossover_method uniform --local_method shc --fig_size 20 --run 10
#python GA.py --dataset facebook --num_label 12 --pop_size 400  --generation 300 --use_local True --selection_method ts --crossover_method uniform --local_method shc --fig_size 20 --run 10


# 2. Tournament -> uniform -> Mutation -> Shc   => Local : False
python GA.py --dataset karate --num_label 2 --pop_size 100 --generation 50 --use_local False --selection_method ts --crossover_method uniform --local_method shc --fig_size 20 --run 10
python GA.py --dataset dolphins --num_label 5 --pop_size 400 --generation 100 --use_local False --selection_method ts --crossover_method uniform --local_method shc --fig_size 20 --run 10
python GA.py --dataset book --num_label 3 --pop_size 400 --generation 100 --use_local False --selection_method ts --crossover_method uniform --local_method shc --fig_size 20 --run 10
python GA.py --dataset football --num_label 12 --pop_size 500 --generation 100 --use_local False --selection_method ts --crossover_method uniform --local_method shc --fig_size 20 --run 10
python GA.py --dataset jazz --num_label 4 --pop_size 400  --generation 200 --use_local False --selection_method ts --crossover_method uniform --local_method shc --fig_size 20 --run 10
#python GA.py --dataset facebook --num_label 12 --pop_size 400  --generation 300 --use_local True --selection_method ts --crossover_method uniform --local_method shc --fig_size 20 --run 10
