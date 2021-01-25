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
from matplotlib.colors import ListedColormap


def main():
    """ Main funciton """

    enc = MHFPEncoder(1024)
    lf = tm.LSHForest(1024, 64)

    fps = []
    hac = []
    c_frac = []
    ring_atom_frac = []
    largest_ring_size = []
    error = []
    error_mol = []
    df = pd.read_csv("/home/boittier/Documents/Tautomers/databases/ANI1_cleaned.csv")
    data = []

    missing = open("/home/boittier/Documents/Tautomers/ANI_missing.txt")
    missing_substructures = [AllChem.MolFromSmiles(x.strip()) for x in missing]
    print(len(missing_substructures))
    missing_substructures = [x for x in missing_substructures if x is not None]
    print(len(missing_substructures))
    missing = []

    ERROR_CUTOFF = 10
    ERROR_CUTOFF_2 = 50

    for i, row in df.iterrows():

        # Mol A
        mol = AllChem.MolFromSmiles(row[2])
        if mol is not None:
            data.append(row[2])
            atoms = mol.GetAtoms()
            size = mol.GetNumHeavyAtoms()
            n_c = 0
            n_ring_atoms = 0
            for atom in atoms:
                if atom.IsInRing():
                    n_ring_atoms += 1
                if atom.GetSymbol().lower() == "c":
                    n_c += 1

            e = abs(row[13])
            if e < ERROR_CUTOFF:
                error.append(e)
            else:
                error.append(ERROR_CUTOFF)

            e = abs(row[7])
            if e < ERROR_CUTOFF_2:
                error_mol.append(e)
            else:
                error_mol.append(ERROR_CUTOFF_2)

            missing_substructure = 0
            for substructure in missing_substructures:
                if mol.HasSubstructMatch(substructure):
                    missing_substructure = 1
            missing.append(missing_substructure)


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

        #  Mol B
        mol = AllChem.MolFromSmiles(row[3])
        if mol is not None:
            data.append(row[3])
            atoms = mol.GetAtoms()
            size = mol.GetNumHeavyAtoms()
            n_c = 0
            n_ring_atoms = 0
            for atom in atoms:
                if atom.IsInRing():
                    n_ring_atoms += 1
                if atom.GetSymbol().lower() == "c":
                    n_c += 1

            e = abs(row[13])
            if e < ERROR_CUTOFF:
                error.append(e)
            else:
                error.append(ERROR_CUTOFF)

            e = abs(row[10])
            if e < ERROR_CUTOFF_2:
                error_mol.append(e)
            else:
                error_mol.append(ERROR_CUTOFF_2)

            missing_substructure = 0
            for substructure in missing_substructures:
                if mol.HasSubstructMatch(substructure):
                    missing_substructure = 1
            missing.append(missing_substructure)


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
    print("Error mol", len(error_mol))
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

    bin_cmap = ListedColormap([ "#2ecc71", "#e74c3c"], name="bin_cmap")
    from matplotlib import cm
    reds_range = cm.get_cmap('Reds', 512)
    newcmp = ListedColormap(reds_range(np.linspace(0.4, 1, 256)))

    f = Faerun(view="front", coords=False)
    f.add_scatter(
        "np_atlas",
        {
            "x": x,
            "y": y,
            "c": [
                error,
                error_mol,
                missing,
                hac,
                c_frak_ranked,
                ring_atom_frac,
                largest_ring_size,
            ],
            "labels": data,
        },
        shader="smoothCircle",
        point_scale=10.0,
        max_point_size=20,
        categorical=[False, False, True, False, False, False, False],
        colormap=[newcmp, "viridis", bin_cmap, "viridis", "viridis", "viridis", "viridis"],
        series_title=[
            "Error (taut)",
            "Error (mol)",
            "Missing",
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
