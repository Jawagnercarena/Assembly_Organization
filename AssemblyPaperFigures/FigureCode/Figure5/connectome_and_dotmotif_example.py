import numpy as np
import networkx as nx
from dotmotif import Motif, GrandIsoExecutor
import pandas
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from scipy.stats import ranksums
import statsmodels.stats.multitest as smm
import v1dd_coregistration_wrangling
 
def calc_ground_truth(pre_cells, synapse, post_cells, use_sizes=False):
    pre_cell_pt_index = pre_cells['pt_root_id']
    post_cell_pt_index = post_cells['pt_root_id']
    synapses = synapse[(synapse['pre_pt_root_id'].isin(pre_cells['pt_root_id'])) & (synapse['post_pt_root_id'].isin(post_cells['pt_root_id']))]
    pre_cell_pt_index = pre_cell_pt_index.to_list()
    post_cell_pt_index = post_cell_pt_index.to_list()

    synapse_connectome = np.zeros((len(pre_cells), len(post_cells)))
    for synapse in tqdm(range(len(synapses))):
        preidx = pre_cell_pt_index.index(synapses.iloc[synapse]['pre_pt_root_id'])
        postidx = post_cell_pt_index.index(synapses.iloc[synapse]['post_pt_root_id'])
        if use_sizes:
            synapse_connectome[preidx,postidx] += synapses.iloc[synapse]['size']
        else:
            synapse_connectome[preidx,postidx] += 1
    return synapse_connectome

tables, mappings = v1dd_coregistration_wrangling.get_tables_and_mappings(online=True)
cell_table = tables['cell']
# Query if you want to select a subset of the cells
# cell_table = cell_table.query('cell_type == "PYC"')[['connectome_index', 'pt_root_id', 'soma_layer']].reset_index()
cell_table['connectome_index'] = cell_table.index

synapse_table = pandas.read_pickle('synapse_Table_668.pickle')
weight_matrix = calc_ground_truth(cell_table, synapse_table, cell_table, use_sizes=True)
adjacency_matrix = weight_matrix.clip(0,1)
pyr_graph = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)

# Motif Analysis with DotMotif
executor = GrandIsoExecutor(graph=pyr_graph)

three_chain = Motif("""
                A -> B
                B -> C
              """)

three_chain_results = executor.find(three_chain)

print('\n\t',len(three_chain_results))
with open('pyc_stefan_motif_three_chain_results.pickle', 'wb') as out_file:
    pickle.dump(three_chain_results, out_file)

four_chain = Motif("""
                A -> B
                B -> C
                C -> D
              """)

four_chain_results = executor.find(four_chain)
print('\n\t',len(four_chain_results))
with open('pyc_stefan_motif_four_chain_results.pickle', 'wb') as out_file:
    pickle.dump(four_chain_results, out_file)