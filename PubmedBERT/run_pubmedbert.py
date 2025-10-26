from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from tqdm import tqdm
import time
import argparse

# Mean Pooling - Take attention mask into account for correct averaging
def meanpooling(output, mask):
    embeddings = output[0] # First element of model_output contains all token embeddings
    mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
    return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


def main(args):
    input_file = args.input_file
    output_csv = args.output_pkl
    
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    elif input_file.endswith('.pkl'):
        df = pd.read_pickle(input_file)   
    else:
        print('Please upload csv or pickle file')
        
    if ("Function" in df.columns) and (output_csv.endswith('.pkl')):
        tokenizer = AutoTokenizer.from_pretrained("./PubmedBERT/weights")
        model = AutoModel.from_pretrained("./PubmedBERT/weights")
        df.reset_index(drop=True, inplace=True)
        df['PubmedBERT'] = None

        empty_rows = df[df['PubmedBERT'].isna()]
        print(f"Total empty:{len(empty_rows)}")
        func_list = empty_rows['Function'].to_list()

        for func in tqdm(func_list):
            inputs = tokenizer([func], padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                output = model(**inputs)
                embeddings = meanpooling(output, inputs['attention_mask'])
            embeddings = embeddings.squeeze(0).cpu().numpy()
            for ind in empty_rows.index:
                df.at[ind, 'PubmedBERT'] = embeddings
            torch.cuda.empty_cache()   
            torch.cuda.synchronize()
            time.sleep(0.01)
        
        print(df['PubmedBERT'][0].shape)  #(768,)
        df.to_pickle(output_csv)
    
    else:
        print("'Function' column must in input csv file, and output file must be pickle format!")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--input_file", type=str, help='input csv or pickle file', required=True)
    argparser.add_argument("-o", "--output_pkl", type=str, help='pkl file', required=True)
    args = argparser.parse_args()    
    main(args)  
