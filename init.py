
import pandas as pd

from rdkit.Chem import Descriptors, QED
from  rdkit import Chem

import jsonlines
import os
import shutil
from tools.docking import calc_affinity
from tools.tools import calculate_sascore,calculate_qed_score
from tools.tools import makedir
from rdkit.Contrib.SA_Score import sascorer
import argparse

parser = argparse.ArgumentParser(description='ELILLM')
parser.add_argument('--exp-path', type=str, default='7L1G_new')
parser.add_argument('--protein-name', type=str, default='7L1G')
parser.add_argument('--threshold', type=float, default=-6.5)
# parser.add_argument('--cfg-smina', type=str, default='config_smina_PRMT5.yaml')
parser.add_argument('--cfg-smina', type=str, default='config_smina_PRMT5.yaml')
args = parser.parse_args()

def main(exp_path, protein_name, dataset='pdb_bind'):
    # import ruamel.yaml as yaml
    # from easydict import EasyDict
    # with open(args.cfg_smina) as f:
    #     cfg_smina = yaml.load(f, Loader=yaml.Loader)
    # args.cfg_smina = EasyDict(cfg_smina)
    args.cfg_smina = None
    smina_path=os.path.join(exp_path,'smina')
    makedir(smina_path)
    # exp_path=args.exp_path
    # protein_name=args.protein_name
    makedir(exp_path)
    data=pd.read_csv(os.path.join(exp_path,'init.csv') )
    smiles=list(data.iloc[:,0])
    docking_scores=[]
    mw_scores=[]
    logp_scores=[]
    qed_scores=[]
    sa_scores=[]
    docking_threshold =args.threshold
    mw_threshold = 700
    logp_threshold = 6.0
    labels=[]
    smiles_filted=[]
    new_target_molecules=[]
    for smile in smiles:
        score=calc_affinity(smile,dir_out=smina_path,name_protein=protein_name,cfg=args.cfg_smina,dataset=dataset)
        print(f"Docking Score: {score}")
        if score==500:
            continue
        docking_scores.append(score)
        smiles_filted.append(smile)
        qed_scores.append(calculate_qed_score(smile))
        mw_scores.append(Descriptors.MolWt(Chem.MolFromSmiles(smile)))
        logp_scores.append(Descriptors.MolLogP(Chem.MolFromSmiles(smile)))
        sa_scores.append(calculate_sascore(smile))

        # radscore_threshold= np.percentile(rad_scores, 75)
    for mol, docking_score, qed_score ,mw,logp, sa in zip(smiles_filted, docking_scores, qed_scores,mw_scores,logp_scores, sa_scores):
        if (
            docking_score <= docking_threshold
            and mw <= mw_threshold
            and logp <= logp_threshold
        ):
            labels.append('1')
            new_target_molecules.append({'smiles': mol, 'label': '1','score':str(docking_score)})
        else:
            labels.append('0')
            new_target_molecules.append({'smiles': mol, 'label': '0','score':str(docking_score)})
    with jsonlines.open(os.path.join(exp_path,'init.jsonl') , mode='a') as writer:
            # writer.write("\\n")
        for molecule in new_target_molecules:
            writer.write(molecule)
            writer.write('\n')
    pd.DataFrame({'smile':smiles_filted,'docking_scores':docking_scores,
                  'QED':qed_scores,'mw':mw_scores,
                  'logp':logp_scores,'SAS':sa_scores,'label':labels}).to_csv(os.path.join(exp_path,'init_score.csv') ,index=False)

def init_pdb_bind():
    dir_path = 'results/pdb_bind_10'
    # protein_list = os.listdir(os.path.join(dir_path, 'random_select'))
    protein_list = ['1o3h', '2vnf', '3lw0', '1ynd', '4c16', '5cyv', '5zty', '6b4h', '7gpb', '8abp']
    # protein_list = ['7gpb', '8abp']
    # dir_path = os.path.join(dir_path, 'init10')
    for protein_name in protein_list:
        exp_path = os.path.join(dir_path, protein_name)
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            shutil.copy(os.path.join(dir_path,'init.csv'), exp_path)
        main(exp_path, protein_name)

def init_cross_docked():
    dir_path = 'results/rand'
    dataset = 'crossdocked'
    for i in range(100):
        protein_name = str(i)
        exp_path = os.path.join(dir_path, protein_name)
        main(exp_path, protein_name, dataset=dataset)

if __name__ == "__main__":
    # exp_path=args.exp_path
    # protein_name=args.protein_name
    init_cross_docked()

    # dir_path = 'pdb_bind/init10'
    # # protein_list = os.listdir(os.path.join(dir_path, 'random_select'))
    # # protein_list = ['1o3h', '2vnf', '3lw0', '1ynd', '4c16', '5cyv', '5zty', '6b4h', '7gpb', '8abp']
    # protein_list = ['7gpb', '8abp']
    # # dir_path = os.path.join(dir_path, 'init10')
    # for protein_name in protein_list:
    #     exp_path = os.path.join(dir_path, protein_name)
    #     if not os.path.exists(exp_path):
    #         os.makedirs(exp_path)
    #         shutil.copy(os.path.join(dir_path,'init.csv'), exp_path)
    #     main(exp_path, protein_name)
