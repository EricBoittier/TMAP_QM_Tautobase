import pickle
import numpy as np
import tmap as tm
import pandas as pd
import scipy.stats as ss
from rdkit.Chem import AllChem
from mhfp.encoder import MHFPEncoder
from faerun import Faerun
from collections import Counter
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt


def main():
    """ Main funciton """

    enc = MHFPEncoder(1024)
    lf = tm.LSHForest(1024, 64)

    fps = []
    hac = []
    c_frac = []
    ring_atom_frac = []
    largest_ring_size = []
    data = open("/home/boittier/Documents/Amons/QM9-smiles.txt").readlines()
    data = [x.strip() for x in data]
    for i, row in enumerate(data):
        print(row)
        if i != 0 and i % 1000 == 0:
            print(100 * i / len(data))
        mol = AllChem.MolFromSmiles(row)
        if mol is not None:
            atoms = mol.GetAtoms()
            size = mol.GetNumHeavyAtoms()
            n_c = 0
            n_ring_atoms = 0
            for atom in atoms:
                if atom.IsInRing():
                    n_ring_atoms += 1
                if atom.GetSymbol().lower() == "c":
                    n_c += 1

            c_frac.append(n_c / size)
            ring_atom_frac.append(n_ring_atoms / size)
            sssr = AllChem.GetSymmSSSR(mol)
            if len(sssr) > 0:
                largest_ring_size.append(max([len(s) for s in sssr]))
            else:
                largest_ring_size.append(0)
            hac.append(size)
            fps.append(tm.VectorUint(enc.encode_mol(mol)))
        else:
            print("failed")

    lf.batch_add(fps)
    print(len(fps))
    lf.index()

    lf.store("lf.dat")
    with open("props.pickle", "wb+") as f:
        pickle.dump(
            (hac, c_frac, ring_atom_frac, largest_ring_size),
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # lf.restore("lf.dat")
    # hac, c_frac, ring_atom_frac, largest_ring_size = pickle.load(
    #     open("props.pickle", "rb")
    # )

    c_frak_ranked = ss.rankdata(np.array(c_frac) / max(c_frac)) / len(c_frac)

    cfg = tm.LayoutConfiguration()
    cfg.node_size = 1 / 26
    cfg.mmm_repeats = 2
    cfg.sl_extra_scaling_steps = 5
    cfg.k = 20
    cfg.sl_scaling_type = tm.RelativeToAvgLength
    x, y, s, t, _ = tm.layout_from_lsh_forest(lf, cfg)

    f = Faerun(view="front", coords=False)
    f.add_scatter(
        "np_atlas",
        {
            "x": x,
            "y": y,
            "c": [
                hac,
                c_frak_ranked,
                ring_atom_frac,
                largest_ring_size,
            ],
            "labels": data,
        },
        shader="smoothCircle",
        point_scale=2.0,
        max_point_size=20,
        categorical=[False, False, False, False],
        colormap=["rainbow", "rainbow", "rainbow", "Blues"],
        series_title=[
            "HAC",
            "C Frac",
            "Ring Atom Frac",
            "Largest Ring Size"
        ],
        has_legend=True,
    )
    f.add_tree("np_atlas_tree", {"from": s, "to": t}, point_helper="np_atlas")
    f.plot(template="smiles")


if __name__ == "__main__":
    main()
