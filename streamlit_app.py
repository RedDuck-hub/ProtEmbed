import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import random
import string
import subprocess
import shutil
import tempfile
import re
import sys
from io import StringIO
import pickle


def generate_random_letters(length=7):
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))
        
def seq_embed():
    with st.sidebar:
        model = st.radio(
        "Choose a embedding model",
        ("ESM2 650M", "ESM Cambrian 600M", "MSA Transformer 100M", "ProtT5 xl_uniref50", "ProTrek 650M")
        )
    st.warning("'Sequence' column must in input file", icon="⚠️")
    upload = st.file_uploader(
    	"Upload prediction file (.csv)", accept_multiple_files=False, type='csv'
    )
    if upload:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(upload.getbuffer())
            real_path = tmp.name
        df = pd.read_csv(upload)
        st.data_editor(df, num_rows="dynamic")
        if "Sequence" in df.columns:
            st.write(f'Choose model: {model}')
            
            if model == "ESM2 650M":
                RUN = "ESM2/run_esm2.py"
                ENV = "esmif"
                random_letters = generate_random_letters()
                OUTPUT_DIR = f'tmp_{random_letters}'
                if st.button('Submit'): 
                    os.mkdir(OUTPUT_DIR)
                    cmd = ['conda', 'run', '-n', ENV, 'python', RUN,
		    '--input_file', str(real_path), '--output_pkl', f"{OUTPUT_DIR}/result.pkl"]
                    with st.status("Running...", expanded=True):
                        result = subprocess.run(cmd,check=True,capture_output=True,text=True)
                    if result.returncode==0:
                        df = pd.read_pickle(f"{OUTPUT_DIR}/result.pkl")
                        df.reset_index(inplace=True, drop=True)
                        st.markdown("**Predict result:**")    
                        st.dataframe(df)
                        shutil.rmtree(OUTPUT_DIR)
                        st.download_button("Press to Download", pickle.dumps(df)   , "result.pkl")  

            elif model == "ESM Cambrian 600M":
                RUN = "ESM_Cambrian/run_esmc.py"
                ENV = "esm3"
                random_letters = generate_random_letters()
                OUTPUT_DIR = f'tmp_{random_letters}'
                if st.button('Submit'): 
                    os.mkdir(OUTPUT_DIR)
                    cmd = ['conda', 'run', '-n', ENV, 'python', RUN,
		    '--input_file', str(real_path), '--output_pkl', f"{OUTPUT_DIR}/result.pkl"]
                    with st.status("Running...", expanded=True):
                        result = subprocess.run(cmd,check=True,capture_output=True,text=True)
                    if result.returncode==0:
                        df = pd.read_pickle(f"{OUTPUT_DIR}/result.pkl")
                        df.reset_index(inplace=True, drop=True)
                        st.markdown("**Predict result:**")    
                        st.dataframe(df)
                        shutil.rmtree(OUTPUT_DIR)
                        st.download_button("Press to Download", pickle.dumps(df)   , "result.pkl")  

            elif model == "MSA Transformer 100M":
                RUN = "MSATransformer/run_msatransformer.py"
                ENV = "esmif"
                random_letters = generate_random_letters()
                OUTPUT_DIR = f'tmp_{random_letters}'
                if st.button('Submit'): 
                    os.mkdir(OUTPUT_DIR)
                    cmd = ['conda', 'run', '-n', ENV, 'python', RUN,
		    '--input_file', str(real_path), '--output_pkl', f"{OUTPUT_DIR}/result.pkl"]
                    with st.status("Running...", expanded=True):
                        result = subprocess.run(cmd,check=True,capture_output=True,text=True)
                    if result.returncode==0:
                        df = pd.read_pickle(f"{OUTPUT_DIR}/result.pkl")
                        df.reset_index(inplace=True, drop=True)
                        st.markdown("**Predict result:**")    
                        st.dataframe(df)
                        shutil.rmtree(OUTPUT_DIR)
                        st.download_button("Press to Download", pickle.dumps(df)   , "result.pkl")  

            elif model == "ProtT5 xl_uniref50":
                RUN = "ProtT5/run_prott5.py"
                ENV = "prott5"
                random_letters = generate_random_letters()
                OUTPUT_DIR = f'tmp_{random_letters}'
                if st.button('Submit'): 
                    os.mkdir(OUTPUT_DIR)
                    cmd = ['conda', 'run', '-n', ENV, 'python', RUN,
		    '--input_file', str(real_path), '--output_pkl', f"{OUTPUT_DIR}/result.pkl"]
                    with st.status("Running...", expanded=True):
                        result = subprocess.run(cmd,check=True,capture_output=True,text=True)
                    if result.returncode==0:
                        df = pd.read_pickle(f"{OUTPUT_DIR}/result.pkl")
                        df.reset_index(inplace=True, drop=True)
                        st.markdown("**Predict result:**")    
                        st.dataframe(df)
                        shutil.rmtree(OUTPUT_DIR)
                        st.download_button("Press to Download", pickle.dumps(df)   , "result.pkl")  

            elif model == "ProTrek 650M":
                RUN = "ProTrek/run_protrek.py"
                ENV = "protrek"
                random_letters = generate_random_letters()
                OUTPUT_DIR = f'tmp_{random_letters}'
                if st.button('Submit'): 
                    os.mkdir(OUTPUT_DIR)
                    cmd = ['conda', 'run', '-n', ENV, 'python', RUN,
		    '--input_file', str(real_path), '--output_pkl', f"{OUTPUT_DIR}/result.pkl", '--embed_choice', 'seq']
                    with st.status("Running...", expanded=True):
                        result = subprocess.run(cmd,check=True,capture_output=True,text=True)
                    if result.returncode==0:
                        df = pd.read_pickle(f"{OUTPUT_DIR}/result.pkl")
                        df.reset_index(inplace=True, drop=True)
                        st.markdown("**Predict result:**")    
                        st.dataframe(df)
                        shutil.rmtree(OUTPUT_DIR)
                        st.download_button("Press to Download", pickle.dumps(df)   , "result.pkl")  
            else:
                pass                                          
        else:
            st.error("'Sequence' column must in dataframe!")     


def struc_embed():
    with st.sidebar:
        model = st.radio(
        "Choose a embedding model",
        ("ESM-IF", "SaProt 650M","ProteinMPNN", "ProTrek 650M")
        )
    st.warning("'Structure' column must in input file", icon="⚠️")
    pdb_dir_path = st.text_input("Absolute path to the PDB folder")
    upload = st.file_uploader(
    	"Upload prediction file (.csv)", accept_multiple_files=False, type='csv'
    )
    if upload:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(upload.getbuffer())
            real_path = tmp.name
        df = pd.read_csv(upload)
        st.data_editor(df, num_rows="dynamic")
        if "Structure" in df.columns:
            st.write(f'Choose model: {model}')

            if model == "ProTrek 650M":
                RUN = "ProTrek/run_protrek.py"
                ENV = "protrek"
                random_letters = generate_random_letters()
                OUTPUT_DIR = f'tmp_{random_letters}'
                if st.button('Submit'): 
                    os.mkdir(OUTPUT_DIR)
                    cmd = ['conda', 'run', '-n', ENV, 'python', RUN,
		    '--input_file', str(real_path), '--output_pkl', f"{OUTPUT_DIR}/result.pkl", '--embed_choice', 'struc', '--pdb_dir', pdb_dir_path]
                    with st.status("Running...", expanded=True):
                        result = subprocess.run(cmd,check=True,capture_output=True,text=True)
                    if result.returncode==0:
                        df = pd.read_pickle(f"{OUTPUT_DIR}/result.pkl")
                        df.reset_index(inplace=True, drop=True)
                        st.markdown("**Predict result:**")    
                        st.dataframe(df)
                        shutil.rmtree(OUTPUT_DIR)
                        st.download_button("Press to Download", pickle.dumps(df)   , "result.pkl")    

            elif model == "ProteinMPNN":
                RUN = "ProteinMPNN/run_pmpnn.py"
                ENV = "pmpnn"
                random_letters = generate_random_letters()
                OUTPUT_DIR = f'tmp_{random_letters}'
                if st.button('Submit'): 
                    os.mkdir(OUTPUT_DIR)
                    cmd = ['conda', 'run', '-n', ENV, 'python', RUN,
		    '--input_file', str(real_path), '--output_pkl', f"{OUTPUT_DIR}/result.pkl", '--pdb_dir', pdb_dir_path]
                    with st.status("Running...", expanded=True):
                        result = subprocess.run(cmd,check=True,capture_output=True,text=True)
                    if result.returncode==0:
                        df = pd.read_pickle(f"{OUTPUT_DIR}/result.pkl")
                        df.reset_index(inplace=True, drop=True)
                        st.markdown("**Predict result:**")    
                        st.dataframe(df)
                        shutil.rmtree(OUTPUT_DIR)
                        st.download_button("Press to Download", pickle.dumps(df)   , "result.pkl")    

            elif model == "ESM-IF":
                RUN = "ESMLF/run_esmlf.py"
                ENV = "esmif"
                random_letters = generate_random_letters()
                OUTPUT_DIR = f'tmp_{random_letters}'
                if st.button('Submit'): 
                    os.mkdir(OUTPUT_DIR)
                    cmd = ['conda', 'run', '-n', ENV, 'python', RUN,
		    '--input_file', str(real_path), '--output_pkl', f"{OUTPUT_DIR}/result.pkl", '--pdb_dir', pdb_dir_path]
                    with st.status("Running...", expanded=True):
                        result = subprocess.run(cmd,check=True,capture_output=True,text=True)
                    if result.returncode==0:
                        df = pd.read_pickle(f"{OUTPUT_DIR}/result.pkl")
                        df.reset_index(inplace=True, drop=True)
                        st.markdown("**Predict result:**")   
                        st.dataframe(df)
                        shutil.rmtree(OUTPUT_DIR) 
                        st.download_button("Press to Download", pickle.dumps(df)   , "result.pkl") 

            elif model == "SaProt 650M":
                RUN = "SaProt/run_saprot.py"
                ENV = "saprot"
                random_letters = generate_random_letters()
                OUTPUT_DIR = f'tmp_{random_letters}'
                if st.button('Submit'): 
                    os.mkdir(OUTPUT_DIR)
                    cmd = ['conda', 'run', '-n', ENV, 'python', RUN,
		    '--input_file', str(real_path), '--output_pkl', f"{OUTPUT_DIR}/result.pkl", '--pdb_dir', pdb_dir_path]
                    with st.status("Running...", expanded=True):
                        result = subprocess.run(cmd,check=True,capture_output=True,text=True)
                    if result.returncode==0:
                        df = pd.read_pickle(f"{OUTPUT_DIR}/result.pkl")
                        df.reset_index(inplace=True, drop=True)
                        st.markdown("**Predict result:**")   
                        st.dataframe(df)
                        shutil.rmtree(OUTPUT_DIR) 
                        st.download_button("Press to Download", pickle.dumps(df)   , "result.pkl")    
                                            
            else:
                pass                                          
        else:
            st.error("'Structure' column must in dataframe!")     
        
        
def msa_embed():
    with st.sidebar:
        model = st.radio(
        "Choose a embedding model",
        ("MSA Pairformer 111M")
        )
    st.warning("'Sequence' and 'MSA' column must in input file", icon="⚠️")
    msa_dir_path = st.text_input("Absolute path to the MSA folder")
    upload = st.file_uploader(
    	"Upload prediction file (.csv)", accept_multiple_files=False, type='csv'
    )
    if upload:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(upload.getbuffer())
            real_path = tmp.name
        df = pd.read_csv(upload)
        st.data_editor(df, num_rows="dynamic")
        if "Sequence" in df.columns and "MSA" in df.columns:
            st.write(f'Choose model: {model}')

            if model == "MSA Pairformer 111M":
                RUN = "MSAPairformer/run_msapairformer.py"
                ENV = "msapairformer"
                random_letters = generate_random_letters()
                OUTPUT_DIR = f'tmp_{random_letters}'
                if st.button('Submit'): 
                    os.mkdir(OUTPUT_DIR)
                    cmd = ['conda', 'run', '-n', ENV, 'python', RUN,
		    '--input_file', str(real_path), '--output_pkl', f"{OUTPUT_DIR}/result.pkl", '--msa_dir', msa_dir_path]
                    with st.status("Running...", expanded=True):
                        result = subprocess.run(cmd,check=True,capture_output=True,text=True)
                    if result.returncode==0:
                        df = pd.read_pickle(f"{OUTPUT_DIR}/result.pkl")
                        df.reset_index(inplace=True, drop=True)
                        #st.markdown("**Predict result:**")    
                        #st.dataframe(df)
                        shutil.rmtree(OUTPUT_DIR)
                        st.download_button("Press to Download", pickle.dumps(df)   , "result.pkl")    
                                            
            else:
                pass                                          
        else:
            st.error("'Sequence' and 'MSA' column must in dataframe!")     
               
def func_embed():
    with st.sidebar:
        model = st.radio(
        "Choose a embedding model",
        ("PubMedBERT base embedding", "ProTrek 650M")
        ) 
    st.warning("'Function' column must in input file", icon="⚠️")
    upload = st.file_uploader(
    	"Upload prediction file (.csv)", accept_multiple_files=False, type='csv'
    )
    if upload:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(upload.getbuffer())
            real_path = tmp.name
        df = pd.read_csv(upload)
        st.data_editor(df, num_rows="dynamic")
        if "Function" in df.columns:
            st.write(f'Choose model: {model}')
            
            if model == "PubMedBERT base embedding":
                RUN = "PubmedBERT/run_pubmedbert.py"
                ENV = "transdw3"
                random_letters = generate_random_letters()
                OUTPUT_DIR = f'tmp_{random_letters}'
                if st.button('Submit'): 
                    os.mkdir(OUTPUT_DIR)
                    cmd = ['conda', 'run', '-n', ENV, 'python', RUN,
		    '--input_file', str(real_path), '--output_pkl', f"{OUTPUT_DIR}/result.pkl"]
                    with st.status("Running...", expanded=True):
                        result = subprocess.run(cmd,check=True,capture_output=True,text=True)
                    if result.returncode==0:
                        df = pd.read_pickle(f"{OUTPUT_DIR}/result.pkl")
                        df.reset_index(inplace=True, drop=True)
                        st.markdown("**Predict result:**")    
                        st.dataframe(df)
                        shutil.rmtree(OUTPUT_DIR)
                        st.download_button("Press to Download", pickle.dumps(df)   , "result.pkl")  

            elif model == "ProTrek 650M":
                RUN = "ProTrek/run_protrek.py"
                ENV = "protrek"
                random_letters = generate_random_letters()
                OUTPUT_DIR = f'tmp_{random_letters}'
                if st.button('Submit'): 
                    os.mkdir(OUTPUT_DIR)
                    cmd = ['conda', 'run', '-n', ENV, 'python', RUN,
		    '--input_file', str(real_path), '--output_pkl', f"{OUTPUT_DIR}/result.pkl", '--embed_choice', 'func']
                    with st.status("Running...", expanded=True):
                        result = subprocess.run(cmd,check=True,capture_output=True,text=True)
                    if result.returncode==0:
                        df = pd.read_pickle(f"{OUTPUT_DIR}/result.pkl")
                        df.reset_index(inplace=True, drop=True)
                        st.markdown("**Predict result:**")    
                        st.dataframe(df)
                        shutil.rmtree(OUTPUT_DIR)
                        st.download_button("Press to Download", pickle.dumps(df)   , "result.pkl")  
            else:
                pass                                          
        else:
            st.error("'Function' column must in dataframe!")     
    

st.header('Protein Multimodal Embedding Generation Platform')
st.write('Supported Modes: Sequence, Structure, MSA and Functional Text')
st.divider()
pg = st.navigation([
    st.Page(seq_embed, title="Sequence embedding"),
    st.Page(struc_embed, title="Structure embedding"),
    st.Page(msa_embed, title="MSA embedding"),
    st.Page(func_embed, title="Function embedding"),
])
pg.run()
