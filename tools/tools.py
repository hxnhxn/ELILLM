import  os
from rdkit.Chem import Descriptors, QED
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit import Chem
from rdkit.Contrib.SA_Score import sascorer
from rdkit.Chem import AllChem, DataStructs,RDKFingerprint
import numpy as np


def get_fingerprint(mols):
    fps = []
    for mol in mols:
        fps.append(RDKFingerprint(mol))
    return fps

def tanimoto_similarity(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def internal_diversity(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    mols = [m for m in mols if m is not None]
    fps = get_fingerprint(mols)
    n = len(mols)
    if n < 2:
        return 0.0
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = tanimoto_similarity(fps[i], fps[j])
            sims.append(sim)
    avg_sim = np.mean(sims)
    return 1 - avg_sim

def makedir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


def calculate_sascore(mol):
    # molecule = Chem.MolFromSmiles(mol)
    # sa_score = rdMolDescriptors.SyntheticAccessibility(molecule)
    try:
        m = Chem.MolFromSmiles(mol)
        sa_score = sascorer.calculateScore(m)

        return (10-sa_score)/10
    except:
        return None
def calculate_qed_score(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is not None:
        qed_score = QED.qed(molecule)
        return qed_score
    else:
        return None