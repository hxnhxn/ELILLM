from tools.docking import calc_affinity
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import SanitizeFlags
if __name__ == '__main__':
    score = 0
    smiles = "FC1=CCC(=S)[C@]23[C](C4C(F)=C(F)C[C]5S[C@]542)C13"
    score = calc_affinity(smiles, dir_out="./", name_protein='14', dataset='crossdocked', output=True)
    mol = Chem.MolFromPDBFile("6_result.pdb")
    if mol is None:
        raise ValueError("无法读取PDB文件")

    # 提取原子坐标
    coordinates = []
    conf = mol.GetConformer()
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        coordinates.append((pos.x, pos.y, pos.z))

    # 打印结果
    print(f"小分子共有 {len(coordinates)} 个原子：")
    for i, (x, y, z) in enumerate(coordinates, 1):
        print(f"原子{i}: X={x:.3f}, Y={y:.3f}, Z={z:.3f}")

    print(score)

    params = AllChem.ETKDGv3()  # 使用 ETKDGv3 参数（推荐）
    params.randomSeed = 42  # 固定随机种子
    params.maxAttempts = 10000  # 最大尝试次数

    mol = Chem.MolFromSmiles(smiles)
    Chem.SanitizeMol(mol, SanitizeFlags.SANITIZE_ALL)
    new_smiles = Chem.MolToSmiles(mol)
    print(smiles)
    print(new_smiles)
    mol = Chem.AddHs(mol)
    status = AllChem.EmbedMolecule(mol, params)
    print(status)