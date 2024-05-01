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
# from networkx.algorithms.bipartite.matrix import biadjacency_matrix as adj

def invert_dict(original_dict):
    inverted_dict = {}
    for key, items in original_dict.items():
        for item in items:
            if item not in inverted_dict:
                inverted_dict[item] = [key]
            else:
                inverted_dict[item].append(key)
    return inverted_dict

def update_dict_keys(key_mapping_dict, dict_to_update):
    return {key_mapping_dict.get(key, key): val for key, val in dict_to_update.items()}

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

import pandas as pd

# Function to find the overlap between two 'assembly' sets in the DataFrame
def find_assembly_overlap(df, first_last_index_list):
    # Ensure the index list has exactly two indexes
    if len(first_last_index_list) != 2:
        return "Please provide a list with exactly two indexes."
    
    # Retrieve the 'assembly' sets for the provided indexes
    try:
        set1 = set(df.at[first_last_index_list[0], 'assembly_membership'])
    except(TypeError):
        return(df.at[first_last_index_list[0], 'assembly_membership'])
    try:
        set2 = set(df.at[first_last_index_list[1], 'assembly_membership'])
    except(TypeError):
        return(df.at[first_last_index_list[1], 'assembly_membership'])

    # Find the intersection (overlap) of the two sets
    overlap = set1.intersection(set2)
    
    return list(overlap)

# Full Matrix
# weight_matrix = np.load('ground_truth_summed_weights_v1dd_668.npy')
# adjacency_matrix = weight_matrix.clip(0,1)
# graph = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)

# Make a graph of just excitatory cells
cell_table_original = pandas.read_feather('pre_cell_table_v1dd_proofread_True_668.feather')
cell_table_original['connectome_index'] = cell_table_original.index
print(len(cell_table_original))
# cell_table = cell_table.query('soma_layer == "23"')[['connectome_index', 'pt_root_id', 'soma_layer']].reset_index()
# cell_table = cell_table_original.query('cell_type == "PYC"')[['connectome_index', 'pt_root_id', 'soma_layer']].reset_index()
cell_table = cell_table_original
print(len(cell_table))
synapse_table = pandas.read_pickle('synapse_Table_668.pickle')
weight_matrix = calc_ground_truth(cell_table, synapse_table, cell_table, use_sizes=True)
adjacency_matrix = weight_matrix.clip(0,1)
# np.save('pyc_test_adjacency_matrix_668.npy', adjacency_matrix)
# adjacency_matrix = np.load('pyc_test_adjacency_matrix_668.npy')
pyr_graph = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
# cell_table.to_feather('pyc_cell_table_index_conn_idx_root_id.feather')

julian_matrix = np.load('ground_truth_connectome_v1dd_1.npy').clip(0,1)

# print("Summed Difference:", np.sum(np.abs((adjacency_matrix - julian_matrix))))
graph_derived_matrix = nx.adjacency_matrix(pyr_graph)
# print("Summed Difference Graph Derived:", np.sum(np.abs((adjacency_matrix - graph_derived_matrix))))
# Motif Analysis with DotMotif
executor = GrandIsoExecutor(graph=pyr_graph)

three_chain = Motif("""
                A -> B
                B -> C
              """)

three_chain_results = executor.find(three_chain)

tables, mappings = v1dd_coregistration_wrangling.get_tables_and_mappings(online=True)
functional_cell_indexes_by_assembly = tables['functional_indexes_by_assembly']
assemblies_by_functional_index = tables['assemblies_by_functional_index']
coregistered_cells = tables['coregistration']
connectome_id_to_root_id_mapping = mappings['connectome_id_to_root_id']
# connectome_id_to_functional_id_mapping = dict(coregistered_cells[['unit_id', 'id']].values)
# functional_id_to_pt_root_id_mapping = dict(coregistered_cells[['id', 'pt_root_id']].values)
# connectome_id_to_root_id_mapping = dict(cell_table[['connectome_index', 'pt_root_id']].values)
assemblies_by_connectome_index = tables['assemblies_by_connectome_id'] #update_dict_keys(connectome_id_to_functional_id_mapping, assemblies_by_functional_index)
coregistered_unit_ids = list(coregistered_cells['unit_id'].values)


# Load assembly and coregistration data, and create useful mappings
# functional_cell_indexes_by_assembly = pandas.read_pickle('v1dd_connectome_cell_indexes_by_assembly.pkl')
# assemblies_by_functional_index = invert_dict(functional_cell_indexes_by_assembly)
# coregistered_cells = pandas.read_feather('scan13_coregistration_dataframe_with_assembly_membership.feather') # Sorted by coregistration ID already
# coregistered_cells['functional_id'] = coregistered_cells['id']
# coregistered_cells['connectome_id'] = coregistered_cells['unit_id']
# cell_table['assembly_membership'] = [np.nan for i in range(len(cell_table))]
# for i in range(len(coregistered_cells)):
#     cell_table[cell_table['pt_root_id'] == coregistered_cells['pt_root_id'][i]]['assembly_membership'] = coregistered_cells['assembly_membership'][i]
# cell_table = cell_table.join(coregistered_cells, how='left', on='pt_root_id', rsuffix='_cor')
# cell_table.set_index('pt_root_id')
# coregistered_cells.set_index('pt_root_id')
# cell_table = cell_table.update(coregistered_cells)
# connectome_id_to_functional_id_mapping = dict(coregistered_cells[['unit_id', 'id']].values)
# functional_id_to_pt_root_id_mapping = dict(coregistered_cells[['id', 'pt_root_id']].values)
# connectome_id_to_root_id_mapping = dict(cell_table[['connectome_index', 'pt_root_id']].values)
# assemblies_by_connectome_index = update_dict_keys(connectome_id_to_functional_id_mapping, assemblies_by_functional_index)
# coregistered_unit_ids = list(coregistered_cells['unit_id'].values)

nonconnected_motifs = 0
motifs_with_origin_and_terminal_coregistered = 0
motifs_without_origin_and_terminal_coregistered = 0

subgraphs_by_assembly = {}
assembly_list = list(functional_cell_indexes_by_assembly.keys())

for assembly_id in assembly_list:
    subgraphs_by_assembly[assembly_id] = nx.Graph()

# for chain in tqdm(three_chain_results):
#     first_last_assemblies = find_assembly_overlap(cell_table, [chain['A'], chain['C']])
#     if np.isnan(first_last_assemblies): # We haven't coregistered it
#        continue 
#     elif first_last_assemblies == []: # We have coregistered it, and it belongs to no assemblies
#         subgraphs_by_assembly['Non-assembly'].add_nodes_from(chain)
#     else: # We have some assemblies
#         for assembly_number in first_last_assemblies:
#             assembly_id = f'Assembly {assembly_number+1}'
#             subgraphs_by_assembly[assembly_id].add_nodes_from(chain)

for chain in tqdm(three_chain_results):
    origin_cell_in_pyc_id, second_cell_in_pyc_id, next_to_last_cell_in_pyc_id, terminal_cell_in_pyc_id = chain['A'], chain['B'], chain['B'], chain['C']

    # Get the connectome (out of all cells, not just PYC) and root IDs for origin, middle, and terminal chain cells
    origin_cell_in_connectome_id = cell_table.iloc[origin_cell_in_pyc_id].connectome_index
    origin_cell_in_root_id = cell_table.iloc[origin_cell_in_pyc_id].pt_root_id

    second_cell_in_connectome_id = cell_table.iloc[second_cell_in_pyc_id].connectome_index
    second_cell_in_root_id = cell_table.iloc[second_cell_in_pyc_id].pt_root_id

    next_to_last_cell_in_connectome_id = cell_table.iloc[next_to_last_cell_in_pyc_id].connectome_index
    next_to_last_cell_in_root_id = cell_table.iloc[next_to_last_cell_in_pyc_id].pt_root_id

    terminal_cell_in_connectome_id = cell_table.iloc[terminal_cell_in_pyc_id].connectome_index
    terminal_cell_in_root_id = cell_table.iloc[terminal_cell_in_pyc_id].pt_root_id

    origin_assembly_ids = None
    terminal_assembly_ids = None
    
    if (origin_cell_in_connectome_id in coregistered_unit_ids) and (terminal_cell_in_connectome_id in coregistered_unit_ids): # If both are coregistered
        motifs_with_origin_and_terminal_coregistered += 1

        if origin_cell_in_connectome_id in tables['assemblies_by_connectome_id'].keys():
            origin_assembly_ids = list(tables['assemblies_by_connectome_id'][origin_cell_in_connectome_id])
        if terminal_cell_in_connectome_id in tables['assemblies_by_connectome_id'].keys():
            terminal_assembly_ids = list(tables['assemblies_by_connectome_id'][terminal_cell_in_connectome_id])

        # We now have origin and terminal assembly IDs.
        psd_sizes_origin_to_mid = list(synapse_table.query('pre_pt_root_id == @origin_cell_in_root_id and post_pt_root_id == @second_cell_in_root_id')['size'].values)
        psd_sizes_mid_to_terminal = list(synapse_table.query('pre_pt_root_id == @next_to_last_cell_in_root_id and post_pt_root_id == @terminal_cell_in_root_id')['size'].values)

        if len(psd_sizes_origin_to_mid) == 0 or len(psd_sizes_mid_to_terminal) == 0: # If we for some reason don't have chains that connect to each other?!
            nonconnected_motifs += 1
            print("\n\tHUGE ERROR!  CHAIN CELLS AREN'T CONNECTED!!!!\n")
        else:
            # Do all combinations of assembly IDs, just for this first option.
            if origin_assembly_ids is not None and terminal_assembly_ids is not None:  # If we can identify assemblies for both
                # Add to each individual matched combo between origin and terminal assemblies
                for temp_origin_assembly_id in origin_assembly_ids:
                    for temp_terminal_assembly_id in terminal_assembly_ids:
                        if temp_origin_assembly_id == temp_terminal_assembly_id: # If they're both in the same assembly
                            # Add to specific assembly
                            subgraphs_by_assembly[temp_origin_assembly_id].add_nodes_from([chain['A'],chain['B'],chain['C']])
                            subgraphs_by_assembly[temp_origin_assembly_id].add_edges_from([(chain['A'],chain['B']), (chain['B'],chain['C'])])
                            # print(f'Assembly {temp_origin_assembly_id+1}', chain)

with open('three_chain_subgraphs_by_assembly.pickle', 'wb') as handle:
    pickle.dump(subgraphs_by_assembly, handle)

# with open('three_chain_subgraphs_by_assembly.pickle', 'rb') as handle:
#     subgraphs_by_assembly = pickle.load(handle)

for assembly_name in assembly_list:
    subgraph = subgraphs_by_assembly[assembly_name]
    print(assembly_name)
    if subgraph.size() > 4:
        print('\tOmega: (0=SW)', nx.omega(subgraph))
        print('\tSigma (1=SW):', nx.sigma(subgraph))
        nx.draw_circular(subgraph)
        plt.savefig(f"subgraphs corrected {assembly_name}.png", format="PNG")
        plt.close()
    else:
        print("Subgraph has too few nodes or edges!")
