import argparse
import os.path
import pandas as pd
from tqdm import tqdm


def main(args):
    import json, time, os, sys, glob
    import shutil
    import warnings
    import numpy as np
    import torch
    from torch import optim
    from torch.utils.data import DataLoader
    from torch.utils.data.dataset import random_split, Subset
    import copy
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    import os.path
    import subprocess
    
    from utils.protein_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB, parse_fasta
    from utils.protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN
    torch.manual_seed(37)
    random.seed(37)
    np.random.seed(37)   
    
    hidden_dim = 128
    num_layers = 3 
    
    input_file = args.input_file
    output_csv = args.output_pkl
    pdb_dir = args.pdb_dir
    
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    elif input_file.endswith('.pkl'):
        df = pd.read_pickle(input_file)   
    else:
        print('Please upload csv or pickle file')
        
    if ("Structure" in df.columns) and (output_csv.endswith('.pkl')):
        df['ProteinMPNN'] = None
        for ind in tqdm(df.index):
            pdb_path = f"{pdb_dir}/{df['Structure'][ind]}"
            
            checkpoint_path = f'ProteinMPNN/vanilla_model_weights/{args.model_name}.pt'    
            alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
            alphabet_dict = dict(zip(alphabet, range(21)))    
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
            pdb_dict_list = parse_PDB(pdb_path, ca_only=False)
            dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=200000)
            all_chain_list = [item[-1:] for item in list(pdb_dict_list[0]) if item[:9]=='seq_chain'] #['A','B', 'C',...]
            #if args.pdb_path_chains:
            #    designed_chain_list = [str(item) for item in args.pdb_path_chains.split()]
            #else:
            designed_chain_list = all_chain_list
            fixed_chain_list = [letter for letter in all_chain_list if letter not in designed_chain_list]
            chain_id_dict = {}
            chain_id_dict[pdb_dict_list[0]['name']]= (designed_chain_list, fixed_chain_list)

            checkpoint = torch.load(checkpoint_path, map_location=device) 
            noise_level_print = checkpoint['noise_level']
            model = ProteinMPNN(ca_only=False, num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=0.00, k_neighbors=checkpoint['num_edges'])
            model.to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            total_residues = 0
            protein_list = []
            total_step = 0
            with torch.no_grad():
                for ix, protein in enumerate(dataset_valid):
                    batch_clones = [copy.deepcopy(protein) for i in range(1)]
                    X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(batch_clones, device, chain_id_dict,ca_only=False)
                    randn_1 = torch.randn(chain_M.shape, device=X.device)
                    #embedding = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1).squeeze(0).cpu().numpy()  
                    embedding = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1).squeeze(0).mean(0).cpu().numpy()  
                    df.at[ind, 'ProteinMPNN'] = embedding
                    torch.cuda.empty_cache()   
                    torch.cuda.synchronize()
                    time.sleep(0.01)
                    
        print(df['ProteinMPNN'][0].shape)  #(128,)
        df.to_pickle(output_csv)
    else:
        print("'Structure' column must in input csv file, and output file must be pickle format!")
        
        
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--model_name", type=str, default="v_48_020", help="ProteinMPNN model name: v_48_002, v_48_010, v_48_020, v_48_030; v_48_010=version with 48 edges 0.10A noise")
    argparser.add_argument("-i", "--input_file", type=str, help='input csv or pickle file', required=True)
    argparser.add_argument("-o", "--output_pkl", type=str, help='pkl file', required=True)
    argparser.add_argument("-d", "--pdb_dir", type=str, help='pdb directory', required=True)
    args = argparser.parse_args()    
    main(args)   
