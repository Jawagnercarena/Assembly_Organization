import numpy as np
import pandas
import standard_transform as st
from tqdm import tqdm
import pathlib
import nglui.statebuilder as sb
import time
import requests

class GetDataset():
    """Get data from online database."""

    def __init__(self, online=False, experiment='microns', resolution=[9., 9., 45], double_layer=False):
        super(GetDataset, self).__init__()

        self.experiment = experiment

        if self.experiment == 'microns' or self.experiment == 'microns_testbed':
            self.resolution = [4.0, 4.0, 40.0]
            self.query_tables = ['allen_v1_column_types_slanted',
                                 'baylor_gnn_cell_type_fine_model_v2', 'aibs_soma_nuc_metamodel_preds_v117']
            # self.query_tables = ['allen_v1_column_types_slanted']
            # # ]
            self.dataset = 'minnie65_phase3_v1'
            self.server_address = None
            self.synapse_table_name = 'synapses_pni_2'
            self.cell_types = ['23P', '4P', '5P-PT', '5P-IT', '5P-NP', '6P-IT', '6P-CT',
                               '6P-U', 'BPC', 'BC', 'MC', 'NGC']
            self.base_cell_types = ['23P', '4P', '5P-PT', '5P-IT', '5P-NP', '6P-IT', '6P-CT',
                               '6P-U', 'BPC', 'BC', 'MC', 'NGC']
            if double_layer:
                self.layers = {
                    '1a': [-20., 43.0], '1b': [43.0, 106.80615154], '23a': [106.80615154, 191.0], '23b': [191.0, 276.21908419], '4a': [276.21908419, 343.5], '4b': [343.5, 411.8631847],
                    '5a': [411.8631847, 482.0], '5b': [482.0, 552.04973966], '6a': [552.04973966, 726.0], '6b': [726.0, 900]}
            else:
                self.layers = {
                    '1': [-20., 106.80615154], '23': [106.80615154, 276.21908419], '4': [276.21908419, 411.8631847],
                    '5': [411.8631847, 552.04973966], '6': [552.04973966, 900]}
            
            self.layer_names = self.layers.keys()
            self.cell_tform = st.minnie_transform_vx()
            self.synapse_tform = st.minnie_transform_vx()

        elif self.experiment == 'v1dd':
            self.resolution = [9., 9., 45]
            self.query_tables = ['manual_central_types']
            self.dataset = 'v1dd'
            self.server_address = 'https://globalv1.em.brain.allentech.org'
            self.synapse_table_name = 'synapses_v1dd'
            self.cell_types = ['PYC', 'BC', 'BPC', 'MC', 'NGC']
            self.base_cell_types = ['PYC', 'BC', 'BPC', 'MC', 'NGC']
            if double_layer:
                self.layers = {
                    '1a': [-20., 40.0], '1b': [40.0, 100.0], '23a': [100.0, 185.0], '23b': [185.0, 270.0], '4a': [270.0, 335.0], '4b': [335.0, 400.0],
                    '5a': [400.0, 475.0], '5b': [475.0, 550.0], '6a': [550.0, 725.0], '6b': [725.0, 900.0]}
            else:
                self.layers = {'1': [-20., 100], '23': [100, 270],
                            '4': [270, 400], '5': [400, 550], '6': [550, 900]}
            self.layer_names = self.layers.keys()
            self.cell_tform = st.v1dd_transform_vx()
            self.synapse_tform = st.v1dd_transform_vx()

        self.layer_bins = np.unique(np.asarray(
            list(self.layers.values())).flatten())

        if online:
            from caveclient import CAVEclient
            import datetime

            if self.experiment == 'microns':
                self.client = CAVEclient(
                    datastack_name=self.dataset, server_address=self.server_address, auth_token="68884fc62f7341c6c6f3d9121affdad0")
                self.timestamp = self.client.materialize.get_version_metadata(version=658)[
                    'time_stamp']
                print("Cave client online and initialized for DB version 658!")
            elif self.experiment == 'v1dd':
                self.client = CAVEclient(
                    datastack_name=self.dataset, server_address=self.server_address, auth_token='c55eb68034e28d0533eef05ead31c8a9')
                self.timestamp = self.client.materialize.get_version_metadata(version=725)[
                    'time_stamp']
                print('Timestamped at Version 725')
                # self.timestamp = datetime.datetime(
                #     2023, 5, 15, 0, 0, 0, 0, tzinfo=datetime.timezone.utc)

            # self.timestamp = self.client.materialize.get_timestamp()
        else:
            print(
                'Operating in Offline Mode.  Any functions called with online=True will cause an error.')

    def which_layer(self, soma_locations):
        if len(soma_locations.shape) == 1:
            soma_locations = np.expand_dims(soma_locations, 0)
        assigned_layers = np.digitize(
            soma_locations[:, 1], self.layer_bins) - 1
        layers = np.asarray(list(self.layers.keys()))[assigned_layers]
        if len(layers) == 1:
            layers = layers[0]
        return layers

    def load_cells(self, proofread=None, only_from_proofread=None, online=False, count=True, override_pickle=None, override_feather=None):
        if online:
            # Get the cell and synapse tables, then filter them for only root IDs associated with proofread axons.
            query_table = self.query_tables[0]
            if self.experiment == 'microns' or self.experiment == 'microns_testbed':
                self.pre_cell_table = self.client.materialize.live_live_query(
                    query_table, desired_resolution=self.resolution, timestamp=self.timestamp)[['pt_root_id', 'cell_type', 'pt_position']]
            elif self.experiment == 'v1dd':
                self.pre_cell_table = self.client.materialize.live_live_query(
                    query_table, desired_resolution=self.resolution, timestamp=self.timestamp, joins=[[query_table, 'target_id', 'nucleus_detection_v0', 'id']])[['pt_root_id', 'target_id', 'cell_type', 'pt_position']]
            # The next step is based on the assumption that the first query table is just the focused column.  If that changes, this will have to change
            self.pre_cell_table['column'] = [
                True for i in range(len(self.pre_cell_table))]
            if len(self.query_tables) > 1:
                for query_table in self.query_tables[1:]:
                    temp = self.client.materialize.live_live_query(
                        query_table, desired_resolution=self.resolution, timestamp=self.timestamp, joins=[[query_table, 'target_id', 'nucleus_detection_v0', 'id']])
                    self.pre_cell_table = self.pre_cell_table.append(
                        temp[['pt_root_id', 'cell_type', 'pt_position']])

            # Do some cleanup
            # Drop dupes
            self.pre_cell_table = self.pre_cell_table.drop_duplicates([
                                                                      'pt_root_id'])

            # Filter out unwanted cell types (non-neuronal, unknown)
            self.pre_cell_table = self.pre_cell_table[np.isin(
                self.pre_cell_table['cell_type'], self.base_cell_types)]

            # Pre-transform all soma locations to flattened cortex micrometer coordinates.
            transformed_positions = self.cell_tform.apply(
                self.pre_cell_table['pt_position'])
            self.pre_cell_table['pt_position'] = transformed_positions

            # Also add soma layer
            pre_cell_layers = self.which_layer(
                np.vstack(self.pre_cell_table['pt_position'].values))
            self.pre_cell_table['soma_layer'] = pre_cell_layers

            if not proofread:  # If the postsynaptic side is not proofread, copy now before we filter the pre, which is always limited to proofread cells.
                self.post_cell_table = self.pre_cell_table.copy()

            self.proofread_root_ids = {}
            if self.experiment == 'microns_testbed':
                proofread_list = np.unique(list(self.client.materialize.live_live_query(
                    'proofreading_status_public_release', timestamp=self.timestamp).query("status_axon!='non'")[
                    'pt_root_id']))
                self.pre_cell_table = self.pre_cell_table[self.pre_cell_table['pt_root_id'].isin(
                    proofread_list)]
            if self.experiment == 'microns':
                proofread_list = np.unique(list(self.client.materialize.live_live_query(
                    'proofreading_status_public_release', timestamp=self.timestamp).query("status_axon == 'extended'")[
                    'pt_root_id']))
                self.pre_cell_table = self.pre_cell_table[self.pre_cell_table['pt_root_id'].isin(
                    proofread_list)]
            if self.experiment == 'v1dd':
                proofread_list = np.unique(list(self.client.materialize.live_live_query(
                    'ariadne_axon_task', timestamp=self.timestamp).query("cell_type == 'submitted'")["pt_root_id"]))
                
                # proofread_list = np.unique(list(self.client.materialize.live_live_query(
                #     'ariadne_axon_task', timestamp=self.timestamp).query("cell_type == 'submitted'")["pt_root_id"]))
                
                
                self.pre_cell_table = self.pre_cell_table[self.pre_cell_table['pt_root_id'].isin(
                    proofread_list)]
            self.proofread_root_ids['pre_pt_root_id'] = self.pre_cell_table['pt_root_id'].values
            self.proofread_root_ids['post_pt_root_id'] = self.pre_cell_table['pt_root_id'].values


            column_idx = []
            idx_counter = 0
            for i in range(len(self.pre_cell_table)):
                if self.pre_cell_table.iloc[i]['column'] == True:
                    column_idx.append(idx_counter)
                    idx_counter += 1
                else:
                    column_idx.append(None)
            self.pre_cell_table['pre_column_idx'] = column_idx

            if proofread:  # If the postsynaptic side is proofread, copy now after filtering the presynaptic
                self.post_cell_table = self.pre_cell_table.copy()
            else:  # If the postsynaptic side is not proofread, then we need to index the column in post_cell_table
                # Note that it will (almost certainly) have different indexes than pre.
                column_idx = []
                idx_counter = 0
                for i in range(len(self.post_cell_table)):
                    if self.post_cell_table.iloc[i]['column'] == True:
                        column_idx.append(idx_counter)
                        idx_counter += 1
                    else:
                        column_idx.append(None)
                self.post_cell_table['post_column_idx'] = column_idx

            # if self.experiment == 'microns_testbed':
            #     self.pre_cell_table = self.pre_cell_table.iloc[0:10]
            #     self.post_cell_table = self.post_cell_table.iloc[0:15]

            self.pre_cell_table.reset_index(drop=True, inplace=True)
            self.post_cell_table.reset_index(drop=True, inplace=True)
        else:  # If Offline
            # print(
            #     f'Loading File: pre_cell_table_{experiment}_proofread_True.h5')
            if override_feather is not None:
                print("Loading from override:", override_feather)
                self.pre_cell_table = pandas.read_feather(override_feather)
                pre_cell_layers = self.which_layer(
                    np.vstack(self.pre_cell_table['pt_position'].values))
                self.pre_cell_table['soma_layer'] = pre_cell_layers
                self.post_cell_table = self.pre_cell_table.copy()
            elif override_pickle is not None:
                print("Loading from override:", override_pickle)
                self.pre_cell_table = pandas.read_pickle(override_pickle)
                pre_cell_layers = self.which_layer(
                    np.vstack(self.pre_cell_table['pt_position'].values))
                self.pre_cell_table['soma_layer'] = pre_cell_layers
                self.post_cell_table = self.pre_cell_table.copy()
            else:
                self.pre_cell_table = pandas.read_feather(
                    f'pre_cell_table_{self.experiment}_proofread_True.feather')
                # self.post_cell_table = self.pre_cell_table
                self.post_cell_table = pandas.read_feather(
                    f'post_cell_table_{self.experiment}_proofread_{proofread}_only_from_proofread_{only_from_proofread}.feather')

            self.proofread_root_ids = {}
            self.proofread_root_ids['pre_pt_root_id'] = self.pre_cell_table['pt_root_id'].values
            # Only the post cells that are also in pre will be proofread
            self.proofread_root_ids['post_pt_root_id'] = self.pre_cell_table['pt_root_id'].values
            self.cell_types = list(self.post_cell_table['cell_type'].unique())

        if count:
            cell_type_list = []
            layer_list = []
            count_list = []
            for cell_type in self.cell_types:
                for layer in self.layers:
                    cells_tmp = self.pre_cell_table[self.pre_cell_table['cell_type'] == cell_type]
                    cells = cells_tmp[cells_tmp['soma_layer']
                                      == layer].reset_index(drop=True)
                    if len(cells) > 0:
                        cell_type_list.append(cell_type)
                        layer_list.append(layer)
                        count_list.append(len(cells))
            print("Presynaptic Cells:")
            [print(f'\"{element[0]};{element[1]};{element[2]}\"', end=" ")
                for element in zip(cell_type_list, layer_list, count_list)]
            print(sum(count_list))

            cell_type_list = []
            layer_list = []
            count_list = []
            for cell_type in self.cell_types:
                for layer in self.layers:
                    cells_tmp = self.post_cell_table[self.post_cell_table['cell_type'] == cell_type]
                    cells = cells_tmp[cells_tmp['soma_layer']
                                      == layer].reset_index(drop=True)
                    if len(cells) > 0:
                        cell_type_list.append(cell_type)
                        layer_list.append(layer)
                        count_list.append(len(cells))
            print("Postsynaptic Cells:")
            [print(f'\"{element[0]};{element[1]};{element[2]}\"', end=" ")
                for element in zip(cell_type_list, layer_list, count_list)]
            print(sum(count_list))

    def load_synapses(self, proofread, only_from_proofread, online=False):
        if online:
            print("Error: Synapse tables are too large to load by online query.  Please call this method with offline=True after creating a synapse file from a data dump.  Speak to the EM team or message the microns consortium to arrange for one if need be.")
        else:
            self.synapse_table = pandas.read_feather(
                f'synapse_table_{self.experiment}_targeting_proofread_{proofread}_only_from_proofread_{only_from_proofread}.feather')
            print(f"Loaded {len(self.synapse_table)} synapses from Disk")
            self.synapse_table = self.synapse_table.drop_duplicates(subset=['id'])
            self.synapse_table = self.synapse_table.reset_index()
            print(f"{len(self.synapse_table)} with duplicates removed")

    def create_cell_files(self, proofread, only_from_proofread, double_layer=False):
        self.load_cells(proofread=proofread, online=True,
                        only_from_proofread=only_from_proofread)

        if self.experiment == 'v1dd':
            updated_cell_types = self.client.materialize.live_live_query(
                'cell_type_skel_features_v1', timestamp=self.timestamp)
            print(self.pre_cell_table.columns)
            print(updated_cell_types.columns)
            self.pre_cell_table = pandas.merge(self.pre_cell_table, updated_cell_types[[
                                            'classification_system', 'cell_type', 'target_id']], how='inner', on='target_id', suffixes=(None, '_new'))
            self.post_cell_table = pandas.merge(self.post_cell_table, updated_cell_types[[
                                                'classification_system', 'cell_type', 'target_id']], how='inner', on='target_id', suffixes=(None, '_new'))
            # print(self.pre_cell_table.columns)
            
            # create a list of cell types to replace
            replace_list = ['PYC']

            # define a function to modify the column
            def modify_cell_type(row):
                # if the fruit value is in the replace list, use the color value instead
                if row['cell_type'] in replace_list:
                    return row['cell_type_new']
                # otherwise, keep the original value
                else:
                    return row['cell_type']

            # apply the function to the two cell dataframes and assign the result to cell type
            self.pre_cell_table['cell_type'] = self.pre_cell_table.apply(modify_cell_type, axis=1)
            self.post_cell_table['cell_type'] = self.post_cell_table.apply(modify_cell_type, axis=1)

            self.pre_cell_table['cell_type'] = self.pre_cell_table['cell_type'].replace('L5', '5P-IT')
            self.pre_cell_table['cell_type'] = self.pre_cell_table['cell_type'].replace(['L2', 'L3'], '23P')
            self.pre_cell_table['cell_type'] = self.pre_cell_table['cell_type'].replace('L4', '4P')
            self.pre_cell_table['cell_type'] = self.pre_cell_table['cell_type'].replace('L5ET', '5P-ET')
            self.pre_cell_table['cell_type'] = self.pre_cell_table['cell_type'].replace('L5NP', '5P-PT')
            self.pre_cell_table['cell_type'] = self.pre_cell_table['cell_type'].replace('L6', '6P')
            
            self.post_cell_table['cell_type'] = self.post_cell_table['cell_type'].replace(['L5'], '5P-IT')
            self.post_cell_table['cell_type'] = self.post_cell_table['cell_type'].replace(['L2', 'L3'], '23P')
            self.post_cell_table['cell_type'] = self.post_cell_table['cell_type'].replace(['L4'], '4P')
            self.post_cell_table['cell_type'] = self.post_cell_table['cell_type'].replace(['L5ET'], '5P-ET')
            self.post_cell_table['cell_type'] = self.post_cell_table['cell_type'].replace(['L5NP'], '5P-PT')
            self.post_cell_table['cell_type'] = self.post_cell_table['cell_type'].replace(['L6'], '6P')

            self.cell_types = list(self.post_cell_table['cell_type'].unique())
            print(self.cell_types)

        elif self.experiment == 'microns':
            self.pre_cell_table['cell_type'] = self.pre_cell_table['cell_type'].replace(['6P-IT', '6P-CT', '6P-U'], '6P')
            self.post_cell_table['cell_type'] = self.post_cell_table['cell_type'].replace(['6P-IT', '6P-CT', '6P-U'], '6P')
            self.cell_types = list(self.post_cell_table['cell_type'].unique())
            print(self.cell_types)

        self.pre_cell_table.sort_values(
            by=['cell_type', 'soma_layer'], inplace=True)
        self.pre_cell_table.reset_index(inplace=True)
        self.post_cell_table.sort_values(
            by=['cell_type', 'soma_layer'], inplace=True)
        self.post_cell_table.reset_index(inplace=True)

        if double_layer:
            experiment_name = self.experiment + '_double_layer'
        else:
            experiment_name = self.experiment

    def create_synapse_file(self, proofread, only_from_proofread):
        self.synapse_table = None
        if self.experiment == 'microns':
            self.synapse_table = pandas.read_csv("mat658_synapses.csv", names=[
                'id', 'ctr_pt_position_x', 'ctr_pt_position_y', 'ctr_pt_position_z', 'pre_pt_root_id', 'post_pt_root_id', 'size'])
            self.synapse_table['ctr_pt_position'] = list(self.synapse_tform.apply((self.synapse_table[[
                'ctr_pt_position_x', 'ctr_pt_position_y', 'ctr_pt_position_z']])))
        elif self.experiment == 'v1dd':
            cell_types = self.cell_types
            layers = self.layers.keys()
            for arbor_type in ['axon', 'dendrite']:
                for cell_type in cell_types:
                    cell_type_filename = cell_type
                    for layer in layers:
                        print(cell_type, layer)
                        if arbor_type == "axon":
                            type_layer_root_ids = self.pre_cell_table[(self.pre_cell_table['cell_type'] == cell_type) & (
                                self.pre_cell_table['soma_layer'] == layer)]['pt_root_id'].values
                        else:
                            type_layer_root_ids = self.post_cell_table[(self.post_cell_table['cell_type'] == cell_type) & (
                                self.post_cell_table['soma_layer'] == layer)]['pt_root_id'].values
                        print(len(type_layer_root_ids), "cells to process")
                        print([int(type_layer_root_ids[i])
                              for i in range(len(type_layer_root_ids))])
                        startingID = 0
                        if self.synapse_table is None:
                            if len(type_layer_root_ids) > 0:
                                while True:
                                    try:
                                        if arbor_type == "axon":
                                            self.synapse_table = self.client.materialize.synapse_query(
                                                pre_ids=int(type_layer_root_ids[0]), timestamp=self.timestamp, desired_resolution=self.resolution)
                                        else:
                                            self.synapse_table = self.client.materialize.synapse_query(
                                                post_ids=int(type_layer_root_ids[0]), timestamp=self.timestamp, desired_resolution=self.resolution)
                                        startingID = 1
                                    except(requests.exceptions.HTTPError):
                                        print(f"HTTP Error: Retrying neuron {type_layer_root_ids[0]}")
                                        print(type_layer_root_ids)
                                        time.sleep(1)
                                        continue
                                    except(FileNotFoundError):
                                        continue
                                    else:
                                        break

                        elif len(type_layer_root_ids) > 0:
                            for root_id in tqdm(type_layer_root_ids[startingID:]):
                                time.sleep(1)
                                retries = 0
                                while retries <= 1:
                                    try:
                                        if arbor_type == "axon":
                                            tmp = self.client.materialize.synapse_query(
                                                pre_ids=[int(root_id)], timestamp=self.timestamp, desired_resolution=self.resolution)
                                        else:
                                            tmp = self.client.materialize.synapse_query(
                                                post_ids=[int(root_id)], timestamp=self.timestamp, desired_resolution=self.resolution)
                                        for key in tmp.attrs.keys():
                                            tmp.attrs[key] = self.synapse_table.attrs[key]
                                        self.synapse_table = pandas.concat(
                                            [self.synapse_table, tmp])
                                    except(requests.exceptions.HTTPError):
                                        print(
                                            "HTTP Error: Retrying neuron", int(root_id))
                                        time.sleep(1)
                                        retries += 1
                                        continue
                                    else:
                                        break

        # pre-transform all synapse locations to flattened cortex micrometer coordinates
        self.synapse_table = self.synapse_table[[
            'id', 'pre_pt_root_id', 'post_pt_root_id', 'size', 'ctr_pt_position']]
        self.synapse_table = self.synapse_table.drop_duplicates(subset=['id'], keep=False)
        self.synapse_table['ctr_pt_position'] = list(
            self.synapse_tform.apply(self.synapse_table['ctr_pt_position']))

        # Remove Autapses
        self.synapse_table = self.synapse_table.drop(
            self.synapse_table[self.synapse_table['pre_pt_root_id'] == self.synapse_table['post_pt_root_id']].index)
        
        # Remove synapses not targeting any of our postsynaptic cells
        self.synapse_table = self.synapse_table.loc[
            self.synapse_table['post_pt_root_id'].isin(self.post_cell_table['pt_root_id'])]
        if only_from_proofread:
            self.synapse_table = self.synapse_table.loc[self.synapse_table['pre_pt_root_id'].isin(
                self.proofread_root_ids['pre_pt_root_id'])].reset_index(drop=True)
        else:
            self.synapse_table.reset_index(drop=True, inplace=True)

        # self.synapse_table.to_feather(
        #     f'synapse_table_{self.experiment}_targeting_proofread_{proofread}_only_from_proofread_{only_from_proofread}.feather')
        # print(f'Saved {len(self.synapse_table)} synapses to disk')

    def save_expected_cell_files(self, experiment_name, proofread, only_from_proofread):
        self.expected_pre_cell_table = pandas.DataFrame()
        self.expected_pre_cell_table = pandas.concat([self.expected_pre_cell_table, self.pre_cell_table[(self.pre_cell_table['cell_type'] == '23P') & (self.pre_cell_table['soma_layer'] == '23')]])
        self.expected_pre_cell_table = pandas.concat([self.expected_pre_cell_table, self.pre_cell_table[(self.pre_cell_table['cell_type'] == '4P') & (self.pre_cell_table['soma_layer'] == '4')]])
        self.expected_pre_cell_table = pandas.concat([self.expected_pre_cell_table, self.pre_cell_table[(self.pre_cell_table['cell_type'][0:2] == '5P') & (self.pre_cell_table['soma_layer'] == '5')]])
        self.expected_pre_cell_table = pandas.concat([self.expected_pre_cell_table, self.pre_cell_table[(self.pre_cell_table['cell_type'] == '6P') & (self.pre_cell_table['soma_layer'] == '6')]])
        self.expected_pre_cell_table = pandas.concat([self.expected_pre_cell_table, self.pre_cell_table[(self.pre_cell_table['cell_type'] == 'BC')]])
        self.expected_pre_cell_table = pandas.concat([self.expected_pre_cell_table, self.pre_cell_table[(self.pre_cell_table['cell_type'] == 'BPC')]])
        self.expected_pre_cell_table = pandas.concat([self.expected_pre_cell_table, self.pre_cell_table[(self.pre_cell_table['cell_type'] == 'MC')]])                                                                      
        self.expected_pre_cell_table = pandas.concat([self.expected_pre_cell_table, self.pre_cell_table[(self.pre_cell_table['cell_type'] == 'NGC')]])

        self.expected_post_cell_table = pandas.DataFrame()
        self.expected_post_cell_table = pandas.concat([self.expected_post_cell_table, self.post_cell_table[(self.post_cell_table['cell_type'] == '23P') & (self.post_cell_table['soma_layer'] == '23')]])
        self.expected_post_cell_table = pandas.concat([self.expected_post_cell_table, self.post_cell_table[(self.post_cell_table['cell_type'] == '4P') & (self.post_cell_table['soma_layer'] == '4')]])
        self.expected_post_cell_table = pandas.concat([self.expected_post_cell_table, self.post_cell_table[(self.post_cell_table['cell_type'][0:2] == '5P') & (self.post_cell_table['soma_layer'] == '5')]])
        self.expected_post_cell_table = pandas.concat([self.expected_post_cell_table, self.post_cell_table[(self.post_cell_table['cell_type'] == '6P') & (self.post_cell_table['soma_layer'] == '6')]])
        self.expected_post_cell_table = pandas.concat([self.expected_post_cell_table, self.post_cell_table[(self.post_cell_table['cell_type'] == 'BC')]])
        self.expected_post_cell_table = pandas.concat([self.expected_post_cell_table, self.post_cell_table[(self.post_cell_table['cell_type'] == 'BPC')]])
        self.expected_post_cell_table = pandas.concat([self.expected_post_cell_table, self.post_cell_table[(self.post_cell_table['cell_type'] == 'MC')]])                                                                      
        self.expected_post_cell_table = pandas.concat([self.expected_post_cell_table, self.post_cell_table[(self.post_cell_table['cell_type'] == 'NGC')]])

        # self.expected_pre_cell_table = self.expected_pre_cell_table[(self.pre_cell_table.cell_type == '23P' & self.pre_cell_table.soma_layer != 23)]
        # self.expected_pre_cell_table = self.expected_pre_cell_table[(self.pre_cell_table.cell_type == '4P' & self.pre_cell_table.soma_layer != 4)]
        # self.expected_pre_cell_table = self.expected_pre_cell_table[(self.pre_cell_table.cell_type[0:2] == '5P' & self.pre_cell_table.soma_layer != 5)]
        # self.expected_pre_cell_table = self.expected_pre_cell_table[(self.pre_cell_table.cell_type == '6P' & self.pre_cell_table.soma_layer != 6)]

        self.expected_pre_cell_table.reset_index(inplace=True)
        self.expected_post_cell_table.reset_index(inplace=True)

        self.expected_pre_cell_table.to_feather(
            f'expected_pre_cell_table_{experiment_name}_proofread_True.feather')
        print(f'Saved {len(self.expected_pre_cell_table)} presynaptic cells to disk')
        self.expected_post_cell_table.to_feather(
            f'expected_post_cell_table_{experiment_name}_proofread_{proofread}_only_from_proofread_{only_from_proofread}.feather')
        print(
            f'Saved {len(self.expected_post_cell_table)} postsynaptic cells to disk')


    def save_cell_files(self, experiment_name, proofread, only_from_proofread):
        self.pre_cell_table.to_feather(
            f'pre_cell_table_{experiment_name}_proofread_True.feather')
        print(f'Saved {len(self.pre_cell_table)} presynaptic cells to disk')
        self.post_cell_table.to_feather(
            f'post_cell_table_{experiment_name}_proofread_{proofread}_only_from_proofread_{only_from_proofread}.feather')
        print(
            f'Saved {len(self.post_cell_table)} postsynaptic cells to disk')

    def save_synapse_file(self, experiment_name, proofread, only_from_proofread):
        self.synapse_table.to_feather(f'synapse_table_{experiment_name}_targeting_proofread_{proofread}_only_from_proofread_{only_from_proofread}.feather')
        print(f'Saved {len(self.synapse_table)} synapses to disk')

if __name__ == '__main__':

    double_layer = False
    for experiment in ['v1dd']: #, microns
        get = GetDataset(online=True, experiment=experiment, double_layer=double_layer)
        for proofread in [False]: #, False
            for only_from_proofread in [False]: #, False
                print(experiment, proofread, only_from_proofread)
                get.create_cell_files(proofread=proofread,
                                    only_from_proofread=only_from_proofread, double_layer=double_layer)
                
                get.create_synapse_file(proofread=proofread,
                                        only_from_proofread=only_from_proofread)
                # get.load_cells(proofread=proofread,
                #                     only_from_proofread=only_from_proofread)
                # get.load_synapses(proofread=proofread,
                #                     only_from_proofread=only_from_proofread)
                # get.save_expected_cell_files(experiment, proofread, only_from_proofread)
                get.save_cell_files(experiment, proofread, only_from_proofread)
                get.save_synapse_file(experiment, proofread, only_from_proofread)
