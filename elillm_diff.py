import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from tools.docking import calc_affinity
from tools.tools import makedir
import jsonlines
import time
import pandas as pd
import numpy as np
import random
import json
from rdkit.Chem import Descriptors, QED
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit.Chem import SanitizeFlags
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.error')

# The number of times to generate new molecules and feed them back into the modelcond
import torch
import time
import subprocess
import re
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import RDConfig
from rdkit.Chem import QED
import os
import sys
from rdkit.Contrib.SA_Score import sascorer
# open source
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines.text_generation import Chat
import argparse
from Bayesian.bayesianLLM import BayesianLLM
from tools.tools import calculate_sascore, calculate_qed_score

parser = argparse.ArgumentParser()
parser.add_argument('--max_num_generations', type=int, default=2000)
parser.add_argument('--not_bo', default=False, action='store_true')
parser.add_argument('--epoch_mlp', type=int, default=200)
parser.add_argument('--epoch_gp', type=int, default=100)
parser.add_argument('--lr_mlp', type=float, default=0.001)
parser.add_argument('--lr_gp', type=float, default=0.1)

args = parser.parse_args()
not_bo = args.not_bo
max_num_generations = args.max_num_generations
epoch_mlp = args.epoch_mlp
epoch_gp = args.epoch_gp
lr_mlp = args.lr_mlp
lr_gp = args.lr_gp



def main(seed, exp_path=None, name_protein=None, dataset='pdb_bind'):
    st_time = time.time()
    record_dict = {'LLM_time': 0, 'train_time':0, 'evaluate_time':0, 'init_time':0, 'num_iterations': 0, 'num_new_mol':0}
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    output_file_path = os.path.join(exp_path, str(seed) + '_result.csv')
    mol_output_file_path = os.path.join(exp_path, str(seed) + '_mol_record.csv')
    makedir("smina")

    # Loading Llama-3.1
    checkpoint = "yourpath/Llama-3.1-8B-Instruct/"

    device_num = 0
    device = f"cuda:{device_num}"  # for GPU usage or "cpu" for CPU usage
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
    # Init bayesianLLM
    repairer_prompt = open("prompt/repairer_prompt.txt", "r").read()
    # explorer_prompt = open("prompt/explorer_prompt.txt", "r").read()

    data = []
    unique_molecules = set()


    init_score = pd.read_csv(os.path.join(exp_path,'init_scores.csv'))
    init_docking_score = init_score["rdkit3d_docking_score"].tolist()
    init_smiles = init_score["smiles"].tolist()
    for i in range(len(init_docking_score)):
        if init_docking_score[i] == 1000:
            init_docking_score[i] = 0
            print(f"{init_smiles[i]} 3D embedded Failed, docking score setting to 0")


    bayesianLLM = BayesianLLM(tokenizer,model,device,
                              repairer_prompt=repairer_prompt, epoch_mlp=epoch_mlp, epoch_gp=epoch_gp,lr_mlp=lr_mlp, lr_gp=lr_gp)
    bayesianLLM.init_dataset(init_smiles, init_docking_score)
    bayesianLLM.init_surrogate()
    unique_molecules.update(init_smiles)

    record_dict['init_time'] += time.time() - st_time
    all_molecules = []
    # Generate new molecules with LSBOLLM k times
    for i in range(1, max_num_generations+1):
        #Generating molecules
        print(f"new molecules num: {len(data)}")
        new_molecules = []
        num_perturbation = 5
        print("protein index", name_protein)
        print("iteration", i)

        st_time = time.time()
        if i != 1:
            bayesianLLM.train_surrogate()
            record_dict['train_time'] += time.time() - st_time

        generated_strings = []
        valid_label = []
        generated_source = []

        st_time = time.time()
        # use bo explore in latent space
        perturbation_mols= bayesianLLM.sample(num_perturbation=num_perturbation)
        record_dict['LLM_time'] += time.time() - st_time

        all_molecules_it = perturbation_mols
        all_molecules.append(all_molecules_it)
        generated_strings.extend(all_molecules_it)
        st_time = time.time()
        # check
        for new_mol in perturbation_mols:
            try:
                mol = Chem.MolFromSmiles(new_mol)
                Chem.SanitizeMol(mol, SanitizeFlags.SANITIZE_ALL)
                new_mol = Chem.MolToSmiles(mol)
                valid_label.append(1)
                if mol is not None and mol not in unique_molecules:
                    #sanitized_mol = Chem.SanitizeMol(mol)
                    if new_mol not in unique_molecules:
                        new_molecules.append(new_mol)
                        generated_source.append('repairer')
                        unique_molecules.add(new_mol)
                    else:
                        print(f"{new_mol} is duplicated")

                    #print("new molecules", new_mol)

            except Exception as e:
                print(f"SMILES Parse Error: {new_mol}. ")
                valid_label.append(0)
                continue
        print("new_molecules", new_molecules)

        #calculate docking scores
        docking_scores = []

        for mol in new_molecules:
            #old
            #docking_score = calculate_docking_score(mol)
            docking_score=calc_affinity(mol,dir_out="smina", name_protein=name_protein, dataset=dataset)
            docking_score = float(docking_score)
            print("docking score", docking_score)
            if docking_score==500:
                docking_scores.append(None)
            else:
                docking_scores.append(docking_score)

        bayesianLLM.update(new_molecules, docking_scores)

        qed_scores = []
        for mol in new_molecules:
            qed_score = calculate_qed_score(mol)
            print("QED score", qed_score)
            qed_scores.append(qed_score)

        #calculate property scores
        mw_scores = []
        logp_scores = []
        sascores = []
        for mol in new_molecules:
            # try:
            mw = Descriptors.MolWt(Chem.MolFromSmiles(mol))
            logp = Descriptors.MolLogP(Chem.MolFromSmiles(mol))
            # sas = rdmd.SyntheticAccessibility(mol)
            mw_scores.append(mw)
            logp_scores.append(logp)
            sascore = calculate_sascore(mol)
            print("SA score", sascore)
            sascores.append(sascore)
            # except:
            #     continue
        all_scores = list(zip(new_molecules, docking_scores, qed_scores, mw_scores, logp_scores, sascores, generated_source))
        filtered_all_scores = []

        for score in all_scores:
            flag = False
            if score[1] is not None and score[1] == 1000:
                print(f"{score[0]} 3D embedded failed, skip it")
                flag = True
            if score[3] is not None and score[3] < 100:
                print(f"The molecular weight of {score[0]} is too small, skip it")
            for column in range(len(score)):
                if score[column] is None:
                    flag = True
            if flag:
                continue
            filtered_all_scores.append(score)

        data.extend(filtered_all_scores)
        # print("data", data)
        record_dict['evaluate_time'] += time.time() - st_time
        record_dict['num_iterations'] = i
        if len(data) >= 100:
            break

    record_dict['num_new_mol'] = len(data)
    if len(data) >= 100:
        data = data[:100]
    df = pd.DataFrame(data, columns=['Molecule', 'Docking Score', 'QED Score', 'MW score', 'LogP Score', 'SA Score', 'Generated Source'])
    df.to_csv(output_file_path, index=False)

    df = pd.DataFrame(all_molecules)
    df.to_csv(mol_output_file_path, index=False)

    with open(os.path.join(exp_path, 'record_dict.json'), 'w') as f:
        json.dump(record_dict, f)

    print(f'generation {i}:')
    print('generated molecules:')
    print(data)


if __name__ == "__main__":
    # exp_path='6GCT_filter'
    # name_protein = '6GCT'
    dataset = 'crossdocked'
    seed = 1
    exp_path = 'results/diff'
    for i in range(100):
        name_protein = str(i)
        main(seed, os.path.join(exp_path, name_protein), name_protein, dataset=dataset)



