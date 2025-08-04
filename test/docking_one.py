from tools.docking import calc_affinity


if __name__ == '__main__':

    smiles = "CC1C2CCC3CC(CC4CC(C3)C(F)C(F)=C4C2)C2CC3CCC(CC4=C(F)C(F)C(C3)CC4C2)C2CCCCC1C2"
    dataset = "crossdocked"
    protein = '0'
    docking_score = calc_affinity(smiles, protein)

