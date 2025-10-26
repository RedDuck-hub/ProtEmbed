import pandas as pd
import time
import os
import esm
from tqdm import tqdm
import torch
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    input_file = args.input_file
    output_pkl = args.output_pkl
    
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    elif input_file.endswith('.pkl'):
        df = pd.read_pickle(input_file)   
    else:
        print('Please upload csv or pickle file')
    
    if ("Sequence" in df.columns) and (output_pkl.endswith('.pkl')):
        model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        model = model.to(device)
        tokenizer = alphabet.get_batch_converter()
   
        seqs = list(set(df['Sequence'].to_list()))
        df['MSATransformer'] = None
        for i, seq in tqdm(enumerate(seqs)):
            single_data = [("seq{}".format(i), seq[:1023])]
            _, _, tokens = tokenizer(single_data)
            tokens = tokens.to(device)
            model.eval()
            with torch.no_grad():
                results = model(tokens, repr_layers=[12])
                representation = results["representations"][12].squeeze().mean(0).cpu().numpy()
            target_rows = df[df['Sequence'] == seq]
            for index in target_rows.index:
                df.at[index, "MSATransformer"] = representation
            torch.cuda.empty_cache()   
            torch.cuda.synchronize()
            time.sleep(0.005)
            
        print(df['MSATransformer'][0].shape)  #(768,)
        df.to_pickle(output_pkl)


    else:
        print("'Sequence' column must in input csv file, and output file must be pickle format!")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--input_file", type=str, help='input csv or pickle file', required=True)
    argparser.add_argument("-o", "--output_pkl", type=str, help='pkl file', required=True)
    args = argparser.parse_args()    
    main(args)  
