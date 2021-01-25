"""Create a visualization of the QM9 molecules using tmap."""
import pickle
import numpy as np
import tmap as tm
import scipy.stats as ss
from faerun import Faerun
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from mhfp.encoder import MHFPEncoder

JOB_NAME = "QM9"


def main():
    """Generate the plot."""
    data = open("/home/boittier/Documents/Amons/QM9-smiles.txt").readlines()

    lf = tm.LSHForest(2048, 128)

    vecs = []
    smiles = []
    logps = []
    n_h_donors = []
    n_h_acceptors = []

    enc = MHFPEncoder()

    for smi in data:
        mol = AllChem.MolFromSmiles(smi)
        if mol is not None:
            smiles.append(smi)
            fp = tm.VectorUint(enc.encode_mol(mol, min_radius=0))
            vecs.append(fp)
            logps.append(Descriptors.MolLogP(mol))
            n_h_donors.append(Descriptors.NumHDonors(mol))
            n_h_acceptors.append(Descriptors.NumHAcceptors(mol))

    lf.batch_add(vecs)
    lf.index()
    lf.store("{}.dat".format(JOB_NAME))

    cfg = tm.LayoutConfiguration()
    cfg.mmm_repeats = 2
    cfg.sl_extra_scaling_steps = 5
    cfg.node_size = 1 / 55

    x, y, s, t, _ = tm.layout_from_lsh_forest(lf, cfg)

    f = Faerun(view="front", coords=False)
    f.add_scatter(
        "QM9",
        {
            "x": x,
            "y": y,
            "c": [logps, n_h_donors, n_h_acceptors],
            "labels": smiles,
        },
        series_title=[
            "cLogP",
            "H Bond Donors",
            "H Bond Acceptors",
        ],
        colormap=["rainbow", "rainbow", "rainbow"],
        has_legend=True,
        max_point_size=20,
        shader="smoothCircle",
    )
    f.add_tree("{}_Tree".format(JOB_NAME), {"from": s, "to": t}, point_helper="{}".format(JOB_NAME))
    f.plot(file_name="{}".format(JOB_NAME), template="smiles")


if __name__ == "__main__":
    main()
