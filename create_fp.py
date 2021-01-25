from mhfp.encoder import MHFPEncoder
from rdkit.Chem import AllChem

enc = MHFPEncoder(512)

with open("DSSTox_0.mhfp", "w+") as f_out:
    with open("DSSTox.smi", "r") as f:
        for line in f.readlines():
            vals = line.split("\t")
            mol = AllChem.MolFromSmiles(vals[0])

            if mol:
                fp = enc.encode_mol(mol, min_radius=0)
                f_out.write(
                    ",".join(map(str, fp)) + "\t" + "\t".join(vals).strip() + "\n"
                )
