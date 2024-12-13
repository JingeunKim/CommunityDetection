# Improving modularity score of community detection using memetic algorithms

This repository contains the code for the research paper titled "Improving modularity score of community detection using memetic algorithms".

# Abstract
With the growth of online networks, understanding the intricate structure of communities has become vital. Traditional community detection algorithms, while effective to an extent, often fall short in complex systems. This study introduced a meta-heuristic approach for community detection
that leveraged a memetic algorithm, combining genetic algorithms (GA) with the stochastic hill climbing (SHC) algorithm as a local optimization method to enhance modularity scores, which was a measure of the strength of community structure within a network. We conducted comprehensive experiments on five social network datasets (Zachary’s Karate Club, Dolphin Social Network, Books About U.S. Politics, American College Football, and the Jazz Club Dataset). Also, we executed an ablation study based on modularity and convergence speed to determine the efficiency of local search. Our method outperformed other GA-based community detection methods, delivering higher maximum and average modularity scores, indicative of a superior detection of community structures. The effectiveness of local search was notable in its ability to accelerate convergence toward the global optimum. Our results not only demonstrated the algorithm's robustness across different network complexities but also underscored the significance of local search in achieving consistent and reliable modularity scores in community detection.



## Usage 
```
python ../main.py\
 --dataset <dataset_name>\
 --num_label <number of label>\
 --pop_size <population size for GA>\
 --generation <number of generations for GA>\
 --use_local <use of local search>\
 --selection_method <selection method for selecting parents>\
 --crossover <crossover methods>\
 --local_method <local search methods>\
 --run <the number of iteration>
```



## Citation
If you find this useful, please cite the following paper:
```
@article{lee2024improving,
  title={Improving modularity score of community detection using memetic algorithms},
  author={Lee, Dongwon and Kim, Jingeun and Yoon, Yourim},
  journal={AIMS Mathematics},
  volume={9},
  number={8},
  pages={20516--20538},
  year={2024}
}
```
