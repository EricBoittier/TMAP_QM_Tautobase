"""Create a visualization of the QM9 molecules using tmap."""
import pickle
import numpy as np
import tmap as tm
import scipy.stats as ss
from faerun import Faerun
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors


def main():
    """Generate the plot."""
    data = pickle.load(open("data.pickle", "rb"))
    lf = tm.LSHForest(512, 32)

    vecs = []
    smiles = []
    logps = []
    c_frac = []
    ring_atom_frac = []
    n_h_donors = []
    n_h_acceptors = []

    for smi, fp in data:
        smiles.append(smi)
        vecs.append(tm.VectorUint(fp))

        mol = AllChem.MolFromSmiles(smi)
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
        logps.append(Descriptors.MolLogP(mol))
        n_h_donors.append(Descriptors.NumHDonors(mol))
        n_h_acceptors.append(Descriptors.NumHAcceptors(mol))

    lf.batch_add(vecs)
    lf.index()
    lf.store("lf.dat")

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
            "c": [logps, c_frac, ring_atom_frac, n_h_donors, n_h_acceptors],
            "labels": smiles,
        },
        series_title=[
            "cLogP",
            "C Fraction",
            "Ring Atom Fraction",
            "H Bond Donors",
            "H Bond Acceptors",
        ],
        colormap=["rainbow", "rainbow", "rainbow", "rainbow", "rainbow"],
        has_legend=True,
        max_point_size=20,
        shader="smoothCircle",
    )
    f.add_tree("QM9_Tree", {"from": s, "to": t}, point_helper="QM9")
    f.plot(template="smiles")


if __name__ == "__main__":
    main()
