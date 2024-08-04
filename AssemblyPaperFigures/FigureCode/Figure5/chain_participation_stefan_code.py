import numpy as np
import networkx as nx
from dotmotif import Motif, GrandIsoExecutor
import matplotlib.pyplot as plt
from scipy.stats import ranksums
import statsmodels.stats.multitest as smm
from data_management import LSMMData
import json
from tqdm import tqdm

with open('chains.json') as f:
    loaded_json = json.load(f)
my_data = LSMMData.LSMMData(loaded_json)
tables = my_data.data
params = my_data.params
dirs = my_data.dirs
mappings = my_data.mappings
 
# # Make a graph of just excitatory cells
# cell_table = tables['structural']['pre_cell']
# cell_table['connectome_index'] = cell_table.index
# synapse_table = tables['structural']['synapse']
# adjacency_matrix = tables['structural']['binary_connectome']
# pyr_graph = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)

# # Motif Analysis with DotMotif
# executor = GrandIsoExecutor(graph=pyr_graph)

# chain_defs = Motif("""
#                 A -> B
#                 B -> C
#                 C -> D
#               """)

# chain_results = executor.find(chain_defs)

# chain_results_array = np.array([list(c.values()) for c in tqdm(chain_results)])
# np.save('all_cell_four_chain_results_array.npy', chain_results_array)
# np.save('all_cell_four_chain_results.npy', chain_results)

chain_results_array = np.load('pyr_cell_four_chain_results_array.npy')

coregistered_cell_indexes = mappings['assemblies_by_connectome_index'].keys()
no_a_cell_indexes = mappings['connectome_indexes_by_assembly']['No A']

# Get chain participation
chain_participation_by_coregistered_cell = {}
for index in tqdm(coregistered_cell_indexes):
    # this only works because the index cannot appear in the same chain more than once
    chain_participation_by_coregistered_cell[index] = np.where(chain_results_array == index)[0].size

pooled_assembly_indexes = list(set(coregistered_cell_indexes) - set(no_a_cell_indexes))
individual_assembly_indexes = [mappings['connectome_indexes_by_assembly'][f'A {i}'] for i in range(1,14)]

# Pool cells which are in assemblies
pooled_assembly_cell_participation = []
for index in tqdm(pooled_assembly_indexes):
    pooled_assembly_cell_participation.append(chain_participation_by_coregistered_cell[index])

# Pool cells which are not in assemblies
no_a_cell_participation = []
for index in tqdm(no_a_cell_indexes):
    no_a_cell_participation.append(chain_participation_by_coregistered_cell[index])

# Pool individual assembly participation
per_assembly_cell_participation = []
for a in range(len(individual_assembly_indexes)):
    per_assembly_cell_participation.append([])
    for index in tqdm(individual_assembly_indexes[a]):
        per_assembly_cell_participation[a].append(chain_participation_by_coregistered_cell[index])
    plt.figure()
    plt.box
    plt.boxplot([per_assembly_cell_participation[a], no_a_cell_participation])
    plt.savefig(f'per_assembly_chain_participation_assembly_{a+1}.png')

# estimate sample size via power analysis
from statsmodels.stats.power import tt_ind_solve_power

# parameters for power analysis
nobs1_array = [len(individual_assembly_indexes[i]) for i in range(len(individual_assembly_indexes))]
mean_no_a = np.mean(no_a_cell_participation)
mean_diff_by_difference = [np.mean(per_assembly_cell_participation[a]) - mean_no_a for a in range(len(per_assembly_cell_participation))]
alpha = 0.05
# perform power analysis
for i, n in enumerate(nobs1_array):
    effect = mean_diff_by_difference[i] / np.std(per_assembly_cell_participation[i])
    r =  np.sum(nobs1_array) / n
    result = tt_ind_solve_power(effect_size = effect, nobs1 = None, alpha = alpha, power = 0.70,  ratio = r)
    print(f'Sample Size for Assembly {i + 1}: %.3f' % result)

plt.figure()
plt.box
plt.boxplot([pooled_assembly_cell_participation, no_a_cell_participation])
plt.savefig('all_cell_4chain_participation.png')

print('A', pooled_assembly_cell_participation)
print('No A', no_a_cell_participation)
print('Done')
