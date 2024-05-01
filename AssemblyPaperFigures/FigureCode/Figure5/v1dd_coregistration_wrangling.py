import numpy as np
import networkx as nx
from dotmotif import Motif, GrandIsoExecutor
import pandas
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from scipy.stats import ranksums
import scipy
import statsmodels.stats.multitest as smm

def which_layer(soma_locations, layers, layer_bins):
    if len(soma_locations.shape) == 1:
        soma_locations = np.expand_dims(soma_locations, 0)
    assigned_layers = np.digitize(
        soma_locations[:, 1], layer_bins) - 1
    layers = np.asarray(list(layers.keys()))[assigned_layers]
    if len(layers) == 1:
        layers = layers[0]
    return layers

def get_cell_and_coregistration_table():
    import standard_transform as st
    from caveclient import CAVEclient
    import datetime

    resolution = [9., 9., 45]
    query_table = 'manual_central_types'
    dataset = 'v1dd'
    server_address = 'https://globalv1.em.brain.allentech.org'
    synapse_table_name = 'synapses_v1dd'
    cell_types = ['PYC', 'BC', 'BPC', 'MC', 'NGC']
    base_cell_types = ['PYC', 'BC', 'BPC', 'MC', 'NGC']
    layers = {'1': [-20., 100], '23': [100, 270],
            '4': [270, 400], '5': [400, 550], '6': [550, 900]}
    layer_names = layers.keys()
    cell_tform = st.v1dd_transform_vx()
    synapse_tform = st.v1dd_transform_vx()
    layer_bins = np.unique(np.asarray(list(layers.values())).flatten())

    client = CAVEclient(
        datastack_name=dataset, server_address=server_address, auth_token='c55eb68034e28d0533eef05ead31c8a9')
    timestamp = client.materialize.get_version_metadata(version=702)[
        'time_stamp']
    
    pre_cell_table = client.materialize.live_live_query(query_table, desired_resolution=resolution, timestamp=timestamp, joins=[[query_table, 'target_id', 'nucleus_detection_v0', 'id']])[['pt_root_id', 'target_id', 'cell_type', 'pt_position']]
    coregistration_table = client.materialize.live_live_query('coregistration_landmarks', desired_resolution=resolution, timestamp=timestamp, joins=[['coregistration_landmarks', 'target_id', 'nucleus_detection_v0', 'id']])[['pt_root_id', 'target_id', 'session', 'scan_idx', 'field', 'unit_id']]

    # Do some cleanup
    # query 'functional_coregistration_v1' if you want to use automated coregistration.  Also add 'residual', 'score' to the indexes.  The below filter isn't for manual.
    # Filter coregistration with a query here: picked these numbers to get about the top 50%, given that the coregistration seems to have roughly a 50% success rate
    # coregistration_table = coregistration_table.query('residual > 9000 and score < 30')

    # Drop dupes
    pre_cell_table = pre_cell_table.drop_duplicates(['pt_root_id'])

    # Filter out unwanted cell types (non-neuronal, unknown)
    pre_cell_table = pre_cell_table[np.isin(
        pre_cell_table['cell_type'], base_cell_types)]

    # Pre-transform all soma locations to flattened cortex micrometer coordinates.
    transformed_positions = cell_tform.apply(
        pre_cell_table['pt_position'])
    pre_cell_table['pt_position'] = transformed_positions

    # Also add soma layer
    pre_cell_layers = which_layer(np.vstack(pre_cell_table['pt_position'].values), layers, layer_bins)
    pre_cell_table['soma_layer'] = pre_cell_layers

    # Restrict to proofread
    proofread_list = np.unique(list(client.materialize.live_live_query(
                    'ariadne_axon_task', timestamp=timestamp).query("cell_type == 'submitted'")["pt_root_id"]))      
    pre_cell_table = pre_cell_table[pre_cell_table['pt_root_id'].isin(proofread_list)]

    return(pre_cell_table, coregistration_table)


def invert_dict(original_dict):
    inverted_dict = {}
    for key, items in original_dict.items():
        for item in items:
            if item not in inverted_dict:
                inverted_dict[item] = [key]
            else:
                inverted_dict[item].append(key)
    return inverted_dict

def map_dict_keys(key_mapping_dict, dict_to_update):
    updated_dict = {}
    for key, val in dict_to_update.items():
        new_key = key_mapping_dict.get(key)
        if new_key is None:
            continue
        else:
            updated_dict[new_key] = val
    return updated_dict #{key_mapping_dict.get(key): val for key, val in dict_to_update.items()}

def get_tables_and_mappings(online=True, include_assemblies=True):
    # Load cell table and add full connectome ID column
    if online:
        cell_table, coregistration_table = get_cell_and_coregistration_table()
        cell_table = cell_table.reset_index()
    else:
        cell_table = pandas.read_feather('pre_cell_table_v1dd_proofread_True_668.feather')
    cell_table['full_connectome_index'] = cell_table.index

    if include_assemblies:
        # Load assembly and coregistration data, and create useful mappings
        v1DD_session13_SGC_ASSEMBLIES = scipy.io.loadmat("./esteps_150000_affinity_04_sessionM409828_13_SGC-ASSEMBLIES.mat", struct_as_record=True, squeeze_me=True)
        ### JULIAN EDIT: REORDER ASSEMBLIES
        ordered_assemblies = sorted(v1DD_session13_SGC_ASSEMBLIES["assemblies"], key = len)
        ordered_assemblies.reverse()
        v1DD_session13_SGC_ASSEMBLIES['assemblies'] = ordered_assemblies

        calcium_fluorescence = np.load('sessionM409828_13_CALCIUM-FLUORESCENCE.npy')

        # Get the functional indexes of neurons involved in each assembly
        # STEFAN REWORK OF JULIAN EDIT: Compose assembly names one-indexed, and zero-index the assembly functional indexes, which start at one in the matlab file
        functional_indexes_by_assembly = {}
        all_assembly_functional_indexes = []
        for assembly_idx in range(len(v1DD_session13_SGC_ASSEMBLIES['assemblies'])):
            one_indexed_temp = list(np.subtract(v1DD_session13_SGC_ASSEMBLIES['assemblies'][assembly_idx], 1))
            functional_indexes_by_assembly[f'A {assembly_idx+1}'] = one_indexed_temp
            all_assembly_functional_indexes += one_indexed_temp
        functional_indexes_by_assembly['No A'] = list(set(range(calcium_fluorescence.shape[1])) - set(all_assembly_functional_indexes))
        assemblies_by_functional_index = invert_dict(functional_indexes_by_assembly)

    # Construct a mapping from the functional index to the roi_id
    roi_id_temp = pandas.DataFrame(np.load('sessionM409828_13_roi_id_by_index.npy'))
    roi_id_temp['functional_index'] = roi_id_temp.index
    roi_id_temp['roi_id'] = roi_id_temp[0]
    roi_id_temp['roi_id'] = roi_id_temp['roi_id'].apply(lambda x: x[0:5] + str(int(x[5])+1) + x[6:])
    functional_index_to_roi_id_mapping = dict(roi_id_temp[['functional_index', 'roi_id']].values)

    if not online:
        # Load Coregistration
        coregistration_table = pandas.read_feather('v1dd_manual_coregistration.feather')

    # Construct ROI IDs and make a mapping to pt_root_ids
    coregistration_table['roi_id'] = [f'plane{coregistration_table["field"].values[i]}_roi_{coregistration_table["unit_id"].values[i]:04d}' for i in range(len(coregistration_table))]
    roi_id_to_pt_root_id_mapping = dict(coregistration_table[['roi_id', 'pt_root_id']].values)

    # Map from root ID to connectome ID, and visa-versa
    connectome_id_to_root_id_mapping = dict(cell_table[['full_connectome_index', 'pt_root_id']].values)
    root_id_to_connectome_id_mapping = dict(cell_table[['pt_root_id', 'full_connectome_index']].values)

    if include_assemblies:
        assemblies_by_roi_id = map_dict_keys(key_mapping_dict=functional_index_to_roi_id_mapping, dict_to_update=assemblies_by_functional_index)
        assemblies_by_pt_root_id = map_dict_keys(key_mapping_dict=roi_id_to_pt_root_id_mapping, dict_to_update=assemblies_by_roi_id)
        assemblies_by_connectome_id = map_dict_keys(key_mapping_dict=root_id_to_connectome_id_mapping, dict_to_update=assemblies_by_pt_root_id)

        mappings = {'pt_root_id_to_connectome_id' : root_id_to_connectome_id_mapping,
                    'connectome_id_to_root_id': connectome_id_to_root_id_mapping,
                    'roi_id_to_pt_root_id': roi_id_to_pt_root_id_mapping,
                    'functional_index_to_roi_id': functional_index_to_roi_id_mapping,
                    }
        
        tables = {'assemblies_by_roi_id' : assemblies_by_roi_id,
                'assemblies_by_pt_root_id' : assemblies_by_pt_root_id,
                'assemblies_by_connectome_id' : assemblies_by_connectome_id,
                'assemblies_by_functional_index' : assemblies_by_functional_index,
                'functional_indexes_by_assembly' : functional_indexes_by_assembly,
                'coregistration' : coregistration_table,
                'cell' : cell_table
                }
    else:
        mappings = {'pt_root_id_to_connectome_id' : root_id_to_connectome_id_mapping,
                    'connectome_id_to_root_id': connectome_id_to_root_id_mapping,
                    'roi_id_to_pt_root_id': roi_id_to_pt_root_id_mapping,
                    'functional_index_to_roi_id': functional_index_to_roi_id_mapping,
                    }
        tables = {'coregistration' : coregistration_table,
                'cell' : cell_table
                }
        # arrays = {'binary_connection_connectome': binary_connection_connectome,
        #           'synapse_count_connectome': synapse_count_connectome,
        #           'PSD_sizes_connectome': PSD_sizes_connectome
        #         }
    return tables, mappings

## Testing code below
# tables, mappings = get_tables_and_mappings(online=True)
# assemblies_by_coregistered = invert_dict(tables['assemblies_by_connectome_id'])
 
# counter = 0
# for val in assemblies_by_coregistered.values():
#     if val != [None]:
#         counter += len(val)
# print("Total Number of Neurons that are Coregistered & Assigned to Assemblies or the None Assemlby Set:", counter)
# for key, val in assemblies_by_coregistered.items():
#     print(key, val)
