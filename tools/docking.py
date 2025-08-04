import os
import subprocess
from loguru import logger
from rdkit import Chem
from rdkit.Chem import AllChem
import time
from easydict import EasyDict
from tqdm import tqdm
import numpy as np


def calc_affinity(sml,dataset='pdb_bind' ,name_protein='6GCT', dir_out='./', prefix='', os_type='linux', num_smina=1, cfg=None, is_del=True, output=False):

    if cfg is not None:
        autobox_add = cfg.autobox_add
        seed = cfg.seed
        exhaustiveness = cfg.exhaustiveness
    else:
        autobox_add = '1'
        seed = '1000'
        exhaustiveness = '16'

    if name_protein == '6GCT':
        file_protein = './datasets/{}_chainA_protein.pdbqt'.format(name_protein)
        file_lig_ref = './datasets/{}_chainA_ligand.pdbqt'.format(name_protein)
    elif name_protein == '8uob':
        file_protein = './datasets/{}_chainA_protein.pdbqt'.format(name_protein)
        file_lig_ref = './datasets/{}_chainA_ligand.pdbqt'.format(name_protein)
        autobox_add = '1'
        seed = '1000'
        exhaustiveness = '16'
    elif name_protein == '7L1G':
        autobox_add = '16'
        file_protein = './datasets/{}_chainA_protein.pdbqt'.format(name_protein)
        file_lig_ref = './datasets/{}_chainA_ligand.pdbqt'.format(name_protein)
    else:
        autobox_add = '1'
        # file_protein = f'./pdb_bind/random_select/{name_protein}/{name_protein}_protein.pdb'
        # file_lig_ref = f'./pdb_bind/random_select/{name_protein}/{name_protein}_pocket.pdb'
        file_protein = f'./datasets/pdb_bind_10/{name_protein}/{name_protein}_protein_processed.pdb'
        file_lig_ref = f'./datasets/pdb_bind_10/{name_protein}/{name_protein}_pocket.pdb'

    if dataset == 'crossdocked':
        autobox_add = '1'
        seed = '1234'
        exhaustiveness = '32'
        file_protein = f'./datasets/crossdocked/structure-files-test/{name_protein}-protein.pdb'
        file_lig_ref = f'./datasets/crossdocked/structure-files-test/{name_protein}-ligand.sdf'

    aff_array = np.zeros(num_smina).astype(np.float32)
    for ii_num in range(num_smina):
        try:
            # smiles to pdb    'BOSearch/exp_BO_Search_seed-1/logs_smina/{}.pdb'.format（time）
            mol = Chem.MolFromSmiles(sml)
            m2 = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 1
            status = AllChem.EmbedMolecule(m2, params)
            if status == -1:
                print("Embedding 3D Molecule Failed")
                return 1000
            # m3 = Chem.RemoveHs(m2)
            AllChem.MMFFOptimizeMolecule(m2)
            file_output = os.path.join(dir_out, prefix + str(time.time()) + '.sdf')
            # Chem.MolToPDBFile(m2, file_output)
            w = Chem.SDWriter(file_output)
            w.write(m2)
            w.close()

            smina_cmd_output = os.path.join(dir_out, prefix + str(time.time()))
            if output:
                launch_args = ['your_smina_path/smina', '-r', file_protein, '-l', file_output, '--autobox_ligand', file_lig_ref,
                               '--autobox_add', autobox_add, '--seed', seed, '--exhaustiveness', exhaustiveness, '-o', prefix + f'{name_protein}_result.pdb',
                               '--cpu', '30', '>>', smina_cmd_output]
            else:
                launch_args = ['your_smina_path/smina', '-r', file_protein, '-l', file_output, '--autobox_ligand', file_lig_ref,
                               '--autobox_add', autobox_add, '--seed', seed, '--exhaustiveness', exhaustiveness, '--cpu','30',
                               '>>', smina_cmd_output]

            #launch_args = ['smina', '-r', file_protein, '-l', file_output,
            #                '--autobox_ligand', file_lig_ref, '--autobox_add', '10',
            #                '--seed', '1000', '--exhaustiveness', '9', '-o', prefix+'dockres.pdb']
            launch_string = ' '.join(launch_args)
            logger.info(launch_string)
            p = subprocess.Popen(launch_string, shell=True, stdout=subprocess.PIPE)

            p.communicate(timeout=1800)




            affinity = 500
            with open(smina_cmd_output, 'r') as f:
                for lines in f.readlines():
                    lines = lines.split()
                    if len(lines) == 4 and lines[0] == '1':
                        affinity = float(lines[1])
            if is_del:
                if 'win' in os_type:
                    prefix_del = 'del '
                else:
                    prefix_del = 'rm -rf '
                p = subprocess.Popen(prefix_del + smina_cmd_output, shell=True, stdout=subprocess.PIPE)
                p.communicate()
                p = subprocess.Popen(prefix_del + file_output, shell=True, stdout=subprocess.PIPE)
                p.communicate()
        except:
            affinity = 500

        if affinity == 500:
            logger.error('**** Affinity error ... ****')
            aff_array = np.array([affinity] * num_smina).astype(np.float32)
            break
        aff_array[ii_num] = affinity

    return aff_array.mean()



