import torch
import re
import pandas as pd
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel 
import argparse
import time


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
        model_path = "ProtT5/prot_t5_xl_uniref50"
        tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False, legacy=True)
        model = T5EncoderModel.from_pretrained(model_path).to(device).eval()

        df.reset_index(drop=True, inplace=True)
        df['ProtT5'] = None
        empty_rows = df[df['ProtT5'].isna()]
        print(f"Total empty:{len(empty_rows)}")
        seq_list = list(set(empty_rows['Sequence'].to_list()))
        for sequence in tqdm(seq_list):    
            sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in [sequence]]
            ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)

            with torch.no_grad():
                embedding_repr = model(input_ids=input_ids,attention_mask=attention_mask)
                emb = embedding_repr.last_hidden_state[0,:len(sequence)].mean(dim=0).cpu().numpy()
        
            target_rows = df[df['Sequence'] == sequence]
            for index in target_rows.index:
                df.at[index, "ProtT5"] = emb
            torch.cuda.empty_cache()   
            torch.cuda.synchronize()
            time.sleep(0.005)
        
        print(df['ProtT5'][0].shape)  #(1024,)
        df.to_pickle(output_pkl)
    
    else:
        print("'Sequence' column must in input csv file, and output file must be pickle format!")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--input_file", type=str, help='input csv or pickle file', required=True)
    argparser.add_argument("-o", "--output_pkl", type=str, help='pkl file', required=True)
    args = argparser.parse_args()    
    main(args)  
