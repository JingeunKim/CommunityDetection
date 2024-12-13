import argparse

import urllib.request
import io
import zipfile
import time
import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
from matplotlib import pyplot
from collections import defaultdict


class GA():
    def __init__(self, G, run):
        self.run = run
        self.generation_size = arg.generation
        self.pop_size = arg.pop_size
        self.columns = len(G.nodes)
        self.num_label = arg.num_label
        # self.tournament_size = 2
        self.mutation_probability = arg.mutation_rate

    def initialize_population(self, graph, population_size, num_label, columns):
        pop = []
        for _ in range(population_size):
            for x in range(len(graph)):
                label = random.randint(0, num_label - 1)
                pop.append(label)
        two_dimensional_list = [pop[i:i + columns] for i in range(0, len(pop), columns)]

        return two_dimensional_list

    def fitness_function(self, individual):
        communities = {key: value for key, value in zip(G.nodes, individual)}
        new_dict = defaultdict(list)

        for key, value in communities.items():
            new_dict[value].append(key)

        result_dict = dict(new_dict)

        modularity = nx.community.modularity(G, result_dict.values())
        return modularity

    def visualize_individual(self, graph, individual):
        # pyplot.rcParams['figure.figsize'] = (arg.fig_size, arg.fig_size)
        # pos = nx.spring_layout(graph)
        # node_colors = [individual[idx] for idx in range(len(graph.nodes()))]
        # print(node_colors)
        # nx.draw(graph, pos, node_color=node_colors, edge_color='gray', cmap=plt.cm.Set1, with_labels=True)


        # Assuming G is your graph and community_colors is a list of colors corresponding to communities
        communities = {key: value for key, value in zip(G.nodes, individual)}
        new_dict = defaultdict(list)

        for key, value in communities.items():
            new_dict[value].append(key)

        result_dict = dict(new_dict)
        # 1. Detect communities using one of NetworkX's built-in algorithms
        communities = result_dict.values()

        # Create a mapping of nodes to their communities
        community_map = {}
        for idx, com in enumerate(communities):
            for node in com:
                community_map[node] = idx

        # 2. Calculate the initial layout positions
        pos = nx.spring_layout(G, k=0.3, iterations=50)

        # 3. Adjust positions based on community (bringing nodes of the same community closer)
        scale = 0.3  # Determines how much closer to the centroid nodes will move (0 - 1)

        for idx, com in enumerate(communities):
            # Calculate the centroid of the community
            centroid = [0, 0]
            for node in com:
                centroid[0] += pos[node][0]
                centroid[1] += pos[node][1]
            centroid = [c / len(com) for c in centroid]

            # Move nodes towards the centroid to group them more tightly
            for node in com:
                pos[node] = [centroid[i] + scale * (pos[node][i] - centroid[i]) for i in range(len(centroid))]

        # 4. Continue with your visualization
        plt.figure(figsize=(12, 12))
        nx.draw_networkx_nodes(G, pos, node_color=[community_map[node] for node in G.nodes()], node_size=250, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=0.7, alpha=0.3)
        node_labels = {node: str(idx) for idx, node in enumerate(G.nodes())}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, alpha=0.7, font_weight='bold')
        plt.axis('off')

        if arg.use_local == "False":
            plt.savefig('./' + arg.dataset + '/' + arg.use_local + '/' + str(self.run) + '_visualize_individual')
        else:
            if arg.local_method == "SA":
                plt.savefig('./' + arg.dataset + '/' + arg.use_local + '/' + arg.local_method + '/' + str(
                    self.run) + '_visualize_individual')
            elif arg.local_method == "shc":
                plt.savefig('./' + arg.dataset + '/' + arg.use_local + '/' + arg.local_method + '/' + str(
                    self.run) + '_visualize_individual')
            elif arg.local_method == "lv":
                plt.savefig('./' + arg.dataset + '/' + arg.use_local + '/' + arg.local_method + '/' + str(
                    self.run) + '_visualize_individual')

        plt.close()

    # Random Selection
    # if arg.selection_method == "rd":
    #     p1, select_idx = self.random_selection(select_idx)
    def random_selection(self, selected_number):
        parents = random.sample(selected_number, 2)
        for rand in parents:
            selected_number.remove(rand)
            return parents, selected_number

    def tournament_selection(self, population, fitness_scores, S_tour=2):
        parents = []
        for _ in range(2):
            tournament = random.sample(list(enumerate(fitness_scores)), S_tour)
            winner_idx = max(tournament, key=lambda item: item[1])[0]
            # print("wi/nner_idx : ", winner_idx)
            parents.append(population[winner_idx])
        # print("fi")
        # Ensure even number of parents for crossover
        if len(parents) % 2 != 0:
            parents.append(random.choice(population))

        return parents

    def uniform_crossover(self, ind1, ind2):
        p1 = ind1
        p2 = ind2
        offspring1 = []
        for i in range(self.columns):
            k = random.random()
            if k >= 0.5:
                offspring1.append(p1[i])
            else:
                offspring1.append(p2[i])

        return offspring1

    def one_way_crossover(self, ind1, ind2):
        p1 = ind1
        p2 = ind2
        offspring1 = []
        pt = random.randint(0, len(ind1) - 1)
        p1_pt = p1[pt]
        for i in range(len(p1)):
            if p1_pt == p1[i]:
                offspring1.append(p1[i])
            else:
                offspring1.append(p2[i])

        return offspring1



    def mutation(self, offspring):
        pt = random.randint(0, len(offspring) - 1)
        offspring[pt] = random.randint(0, self.num_label - 1)
        return offspring



    def draw_GA(self, fitness_best, fitness_avg):
        generations = range(1, self.generation_size + 1)
        fitness_std = np.std(fitness_avg)
        plt.scatter(generations, fitness_best, label='Best Fitness', edgecolors='yellow')
        plt.plot(generations, fitness_avg, label='Average Fitness', alpha=0.5)

        plt.xlabel('Number of generations')
        plt.ylabel('Fitness value')
        plt.legend()
        plt.grid(True)

        if arg.use_local == "False":
            plt.savefig('./' + arg.dataset + '/' + arg.use_local + '/' + str(self.run) + '_GA_fitness_')
        else:
            if arg.local_method == "SA":
                plt.savefig('./' + arg.dataset + '/' + arg.use_local + '/' + arg.local_method + '/' + str(
                    self.run) + '_GA_fitness_')
            elif arg.local_method == "shc":
                plt.savefig('./' + arg.dataset + '/' + arg.use_local + '/' + arg.local_method + '/' + str(
                    self.run) + '_GA_fitness_')
            elif arg.local_method == "lv":
                plt.savefig('./' + arg.dataset + '/' + arg.use_local + '/' + arg.local_method + '/' + str(
                    self.run) + '_GA_fitness_')

        plt.close()

    def lv(self, offspring):
        pass

    def shc(self, offspring):
        current_individual = offspring.copy()
        idx = random.randint(0, len(current_individual) - 1)
        neighbor = current_individual.copy()
        current_fitness = self.fitness_function(offspring)
        change = []
        for p in range(self.num_label):
            neighbor[idx] = p
            change.append(self.change_modularity(neighbor,current_fitness))
        max_idx = change.index(max(change))
        current_individual[idx] = max_idx
        return current_individual

    def change_modularity(self, individual, current_fitness):
        change = self.fitness_function(individual)
        return change - current_fitness

    def SA(self, chromosome, T_initial=100000, k=0.99, I=10):
        import math
        current_temp = T_initial
        current_solution = chromosome
        current_modularity = self.fitness_function(current_solution)
        # print("고치기 전 솔루션")
        # print(current_solution)고
        # print(self.fitness_function(current_solution))
        for _ in range(I):
            new_solution = self.mutation(chromosome)
            new_modularity = self.fitness_function(new_solution)
            p = new_modularity - current_modularity
            # print("p = ", p, " math.exp((-p / current_temp)) = ", math.exp((-p / current_temp)))
            if p > 0 or math.exp(-p) / current_temp > random.random():
                current_solution, current_modularity = new_solution, new_modularity
                # print("고친 솔루션")
                # # print(current_solution)
                # print(current_modularity)
            current_temp *= k
        # print("고친 솔루션")
        # print(current_solution)
        # print(self.fitness_function(current_solution))
        return current_solution

    def NMI(self, individual):
        from sklearn.metrics.cluster import normalized_mutual_info_score
        if arg.dataset == "karate":
            groundTruth = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0,
                           0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            return normalized_mutual_info_score(individual, groundTruth)
        elif arg.dataset == "dolphins":
            groundTruth = [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1,
                           0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1]
            return normalized_mutual_info_score(individual, groundTruth)
        elif arg.dataset == "book":
            groundTruth = [1, 2, 2, 2, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 0,
                           0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]

            return normalized_mutual_info_score(individual, groundTruth)
        elif arg.dataset == "football":
            groundTruth = [8, 1, 0, 5, 8, 5, 0, 9, 9, 8, 5, 4, 11, 0, 11, 0, 8, 7, 11, 2, 7, 9, 9, 8, 4, 1, 11, 7, 6, 2,
                           2, 11, 0, 1, 11, 2, 10, 1, 11, 0, 5, 8, 10, 11, 3, 1, 6, 0, 3, 6, 4, 9, 5, 6, 11, 2, 7, 3, 6,
                           4, 0, 11, 7, 4, 0, 7, 3, 6, 9, 4, 7, 11, 5, 6, 5, 3, 7, 9, 9, 2, 10, 5, 10, 6, 5, 11, 3, 7,
                           6, 1, 10, 3, 3, 8, 2, 7, 7, 4, 5, 11, 0, 2, 5, 1, 8, 1, 0, 5, 9, 1, 3, 9, 3, 7, 6]

            return normalized_mutual_info_score(individual, groundTruth)

    def repair(self, individual, missing_number):
        # print("repair phase")
        # print(missing_number)
        # print(individual)
        perc = self.columns // self.num_label
        for i in range(len(missing_number)):
            lst = list(range(self.columns))
            pt = random.sample(lst, perc)
            for j in pt:
                individual[j] = missing_number[i]
        # print("Done")
        # print(individual)

        return individual

    def evolve(self):
        random.seed(self.run)
        population = self.initialize_population(G, self.pop_size, self.num_label, self.columns)

        fitness = []
        fitness_best = []
        fitness_avg = []
        for i in range(len(population)):
            fitness.append(self.fitness_function(population[i]))

        for generation in range(self.generation_size):
            print("------------------------------------", generation + 1,
                  "generation ------------------------------------")
            select_idx = list(range(len(population)))  # selected_number = list(range(len(new_population)//3))
            for _ in range(self.pop_size // 2):
                # p1 = self.select_parents(population, fitness)
                # offspring1 = self.uniform_crossover(p1[0],p1[1])
                if arg.selection_method == "rd":
                    p1, select_idx = self.random_selection(select_idx)
                    if arg.crossover_method == "uniform":
                        offspring1 = self.uniform_crossover(population[p1[0]], population[p1[1]])
                    elif arg.crossover_method == "ow":
                        offspring1 = self.one_way_crossover(population[p1[0]], population[p1[1]])
                elif arg.selection_method == "ts":
                    p1 = self.tournament_selection(population, fitness)
                    if arg.crossover_method == "uniform":
                        offspring1 = self.uniform_crossover(p1[0], p1[1])
                    elif arg.crossover_method == "ow":
                        offspring1 = self.one_way_crossover(p1[0], p1[1])
                mutation_p = random.random()
                if self.mutation_probability > mutation_p:
                    # print(fitness)
                    # print(idx)
                    # print(population[idx])
                    offspring1 = self.mutation(offspring1)
                if arg.use_local == "True":
                    if arg.local_method == "shc":
                        idx = fitness.index(max(fitness))
                        offspring1 = self.shc(offspring1)
                    elif arg.local_method == "SA":
                        offspring1 = self.SA(offspring1)
                    elif arg.local_method == "lv":
                        # offspring = self.lv(offspring)
                        pass
                elif arg.use_local == "False":
                    pass
                population.append(offspring1)
                # print("offspr fitenss")
                # print(self.fitness_function(offspring))
                # print(offspring1)
                for p in range(self.pop_size):
                    # print(p, "번째 사람")
                    find_missing_number = set(population[p])
                    # print("find_missing_number = ", find_missing_number)
                    missing_number = []
                    for q in range(self.num_label):
                        if q not in find_missing_number:
                            missing_number.append(q)
                            # print(q)
                    if len(missing_number) > 0:
                        population[p] = self.repair(population[p], missing_number)
                fitness.append(self.fitness_function(offspring1))

            fitness_rank = np.argsort(fitness)[::-1]
            population = [population[i] for i in fitness_rank]
            fitness = [fitness[i] for i in fitness_rank]
            population = population[:self.pop_size]
            fitness = fitness[:self.pop_size]
            # for a in population:
            #     print(a)
            print("best fitness = ", fitness[0])
            print("avg fitness = ", sum(fitness) / len(fitness))
            fitness_best.append(fitness[0])
            fitness_avg.append(sum(fitness) / len(fitness))

            # pyplot.rcParams['figure.figsize'] = (arg.fig_size, arg.fig_size)
            # pos = nx.spring_layout(G)
            # best_ = population[0]
            # node_colors = [best_[idx] for idx in range(len(G.nodes()))]
            # print(node_colors)
            # repair = []
            # for p in range(self.pop_size):
            #     for q in range(self.columns):

            # nx.draw(G, pos, node_color=node_colors, edge_color='gray', cmap=plt.cm.Set1, with_labels=True)
            # plt.show()
            # plt.close()

        best_fit = fitness[0]
        print("final best = ", best_fit)
        print(population[0])
        self.visualize_individual(G, population[0])
        self.draw_GA(fitness_best, fitness_avg)


        return best_fit, fitness_avg, population[0]


if __name__ == "__main__":
    import logging
    import datetime

    parser = argparse.ArgumentParser(description="parameter of CDGA")
    parser.add_argument("--dataset", type=str, default="football", help="karate or dolphins or book or football")
    parser.add_argument("--pop_size", type=int, default=100, help="population size")
    parser.add_argument("--num_label", type=int, default=12,
                        help="number of label; karate, dolphins: 2, book: 3, football: 12")
    parser.add_argument("--selection_method", type=str, default="ts",
                        help="rd : random, ts : tournament selection")
    parser.add_argument("--crossover_method", type=str, default="ow",
                        help="uniform : uniform crossover, ow : one-way crossover")
    parser.add_argument("--local_method", type=str, default="shc",
                        help="shc : Stochastic hill climbing, SA : simulate annealing, lv : lauvain algorithm")
    parser.add_argument("--mutation_rate", type=float, default=0.5, help="probability of mutation")
    parser.add_argument("--generation", type=int, default=100, help="number of generation")
    parser.add_argument("--run", type=int, default=1, help="number of runtime")
    parser.add_argument("--fig_size", type=int, default=12, help="size of figure")
    parser.add_argument("--use_local", type=str, default="True", help="Select to use local search or not")
    arg = parser.parse_args()
    print(arg)


    def setup_logger():
        logger = logging.getLogger()
        log_path = './logs/{:%Y%m%d}_{}.log'.format(datetime.datetime.now(), arg.dataset)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')
        fh = logging.FileHandler(log_path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger


    def print_and_log(logger, msg):
        # global logger
        print(msg)
        logger.info(msg)


    logger = setup_logger()

    if arg.dataset == 'karate':
        G = nx.karate_club_graph()
    elif arg.dataset == 'football':
        url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"

        sock = urllib.request.urlopen(url)  # open URL
        s = io.BytesIO(sock.read())  # read into BytesIO "file"
        sock.close()

        zf = zipfile.ZipFile(s)  # zipfile object
        txt = zf.read("football.txt").decode()  # read info file
        gml = zf.read("football.gml").decode()  # read gml data
        # throw away bogus first line with # from mejn files
        gml = gml.split("\n")[1:]
        G = nx.parse_gml(gml)  # parse gml data
    elif arg.dataset == 'book':
        url = "http://www-personal.umich.edu/~mejn/netdata/polbooks.zip"

        sock = urllib.request.urlopen(url)  # open URL
        s = io.BytesIO(sock.read())  # read into BytesIO "file"
        sock.close()

        zf = zipfile.ZipFile(s)  # zipfile object
        txt = zf.read("polbooks.txt").decode()  # read info file
        gml = zf.read("polbooks.gml").decode()  # read gml data
        # throw away bogus first line with # from mejn files
        gml = gml.split("\n")[1:]
        G = nx.parse_gml(gml)  # parse gml data
    elif arg.dataset == 'dolphins':
        url = "http://www-personal.umich.edu/~mejn/netdata/dolphins.zip"

        sock = urllib.request.urlopen(url)  # open URL
        s = io.BytesIO(sock.read())  # read into BytesIO "file"
        sock.close()

        zf = zipfile.ZipFile(s)  # zipfile object
        txt = zf.read("dolphins.txt").decode()  # read info file
        gml = zf.read("dolphins.gml").decode()  # read gml data
        # throw away bogus first line with # from mejn files
        gml = gml.split("\n")[1:]
        G = nx.parse_gml(gml)  # parse gml data


    elif arg.dataset == 'facebook':
        # url = 'https://snap.stanford.edu/data/facebook_combined.txt.gz'
        #
        # sock = urllib.request.urlopen(url)  # open URL
        # s = io.BytesIO(sock.read())  # read into BytesIO "file"
        # sock.close()
        #
        # zf = zipfile.ZipFile(s)  # zipfile object
        # txt = zf.read("facebook_combined.txt").decode()  # read info file
        # # gml = zf.read("dolphins.gml").decode()  # read gml data
        # # # throw away bogus first line with # from mejn files
        # # gml = gml.split("\n")[1:]
        G = nx.read_edgelist('./facebook_combined.txt', create_using =nx.Graph, nodetype=int )

        # print(gml)

    elif arg.dataset == 'jazz':
        file_path = './jazz.net'

        # Initialize an empty graph
        G = nx.Graph()

        # Manually parse the file
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('*Edges') or line.startswith('*Arcs'):
                    # We've reached the edges section; subsequent lines describe edges
                    continue
                if line.strip() and not line.startswith('*'):
                    # Parse an edge line; expected format: "source target weight"
                    source, target, weight = map(int, line.strip().split())
                    G.add_edge(source, target, weight=weight)


    fit = []
    all_run = []
    all_nmi = []
    for i in range(arg.run):
        print("===========================================", i, "===========================================")
        start = time.time()
        # fitness_best, fitness_avg, population, NMI = GA(G, i).evolve()
        fitness_best, fitness_avg, population = GA(G, i).evolve() # With Jazz dataset and Facebook dataset
        end = time.time()
        el_time = end - start
        all_run.append(el_time)
        fit.append(fitness_best)

    idx = fit.index(max(fit))

    print_and_log(logger, "=" * 50)
    print_and_log(logger, arg)
    print_and_log(logger, "finish")
    print_and_log(logger, "dataset : " + str(arg.dataset) + " local search : " + str(
        arg.use_local) + " selection method = " + arg.selection_method + " crossover = " + arg.crossover_method + " local method : " + arg.local_method)
    print_and_log(logger, "all fitness = {}".format(fit))
    print_and_log(logger, "good solution fitness = {}/ index = {}".format(fit[idx], idx))
    print_and_log(logger, "avg fitness = {}, std = {}".format(sum(fit) / len(fit), np.std(fit)))
    # print_and_log(logger, "avg NMI =  {}, std = {}".format(sum(all_nmi) / len(all_nmi), np.std(all_nmi)))
    print_and_log(logger, "avg time =  {}, std = {}".format(sum(all_run) / len(all_run), np.std(all_run)))
    print_and_log(logger, "=" * 50)
