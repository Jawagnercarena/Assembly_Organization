import pandas
import numpy as np

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

cell_table = pandas.read_feather('pre_cell_table_v1dd_proofread_True_668.feather')
cell_table['connectome_index'] = cell_table.index
functional_cell_indexes_by_assembly = pandas.read_pickle('v1dd_connectome_cell_indexes_by_assembly.pkl')
assemblies_by_functional_index = invert_dict(functional_cell_indexes_by_assembly)
coregistered_cells = pandas.read_feather('scan13_coregistration_dataframe_with_assembly_membership.feather') # Sorted by coregistration ID already
connectome_id_to_functional_id_mapping = dict(coregistered_cells[['unit_id', 'id']].values)
functional_id_to_pt_root_id_mapping = dict(coregistered_cells[['id', 'pt_root_id']].values)
connectome_id_to_root_id_mapping = dict(cell_table[['connectome_index', 'pt_root_id']].values)
assemblies_by_connectome_index = update_dict_keys(connectome_id_to_functional_id_mapping, assemblies_by_functional_index)

print(assemblies_by_connectome_index)

