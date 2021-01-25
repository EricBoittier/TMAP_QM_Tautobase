import sys
import pickle
import numpy as np
import pandas as pd
import tmap as tm
import scipy.stats as ss
from faerun import Faerun
from rdkit.Chem import AllChem, Descriptors
from mhfp.encoder import MHFPEncoder
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def main():
    """ The main function """
    lf = tm.LSHForest(512, 32)

    fps = []
    smiles = []
    labels = []
    names = []
    groups = []
    logp = []

    with open("DSSTox_0.mhfp", "r") as f:
        for i, line in enumerate(f):
            if i % 10000 == 0:
                print(i)

            vals = line.strip().split("\t")

            mol = AllChem.MolFromSmiles(vals[1])

            if (
                mol.GetNumAtoms() < 6 and vals[1].count(".") > 1
            ) or mol.GetNumAtoms() < 3:
                continue

            fps.append(tm.VectorUint(list(map(int, vals[0].split(",")))))
            smiles.append(vals[1])

            label = (
                vals[1]
                + '__<a href="https://comptox.epa.gov/dashboard/'
                + vals[3]
                + '">'
                + vals[3]
                + "</a>"
                + "__"
                + vals[4]
            )
            labels.append(label.replace("'", "´"))
            names.append(vals[4])
            groups.append(vals[5])
            logp.append(Descriptors.MolLogP(mol))

    # Sort the thing to get high values on top
    groups, smiles, labels, names, logp, fps = (
        list(t)
        for t in zip(
            *sorted(zip(groups, smiles, labels, names, logp, fps), reverse=True)
        )
    )

    lf.batch_add(fps)
    lf.index()

    lf.store("dsstox.dat")
    with open("labels.dat", "wb+") as f:
        pickle.dump(
            (smiles, labels, names, groups, logp), f, protocol=pickle.HIGHEST_PROTOCOL
        )

    # lf.restore("dsstox.dat")
    # smiles, labels, names, groups, logp = pickle.load(open("labels.dat", "rb"))

    group_mapping = {
        "DSSTox_High": "0 - High (DSSTox)",
        "DSSTox_Low": "3 - Low (DSSTox)",
        "Public_High": "1 - High (Public)",
        "Public_Low": "4 - Low (Public)",
        "Public_Medium": "2 - Medium (Public)",
        "Public_Untrusted": "5 - Untrusted (Public)",
    }
    groups = [group_mapping[g] for g in groups]

    logp_ranked = ss.rankdata(np.array(logp) / max(logp)) / len(logp)
    labels_groups, groups = Faerun.create_categories(groups)

    cfg = tm.LayoutConfiguration()
    cfg.k = 20
    cfg.sl_extra_scaling_steps = 10
    # cfg.sl_repeats = 12
    # cfg.mmm_repeats = 2
    cfg.node_size = 1 / 70

    x, y, s, t, _ = tm.layout_from_lsh_forest(lf, config=cfg)
    x = list(x)
    y = list(y)
    s = list(s)
    t = list(t)
    pickle.dump(
        (x, y, s, t), open("coords.dat", "wb+"), protocol=pickle.HIGHEST_PROTOCOL
    )

    # x, y, s, t = pickle.load(open("coords.dat", "rb"))

    custom_cmap = ListedColormap(
        ["#eb2f06", "#eb2f06", "#fa983a", "#78e08f", "#78e08f", "#4a69bd"],
        name="custom",
    )

    bin_cmap = ListedColormap(["#e74c3c", "#2ecc71"], name="bin_cmap")

    f = Faerun(
        clear_color="#222222",
        coords=False,
        view="front",
        impress='made with <a href="http://tmap.gdb.tools" target="_blank">tmap</a><br />and <a href="https://github.com/reymond-group/faerun-python" target="_blank">faerun</a>',
    )

    f.add_scatter(
        "DSSTox",
        {"x": x, "y": y, "c": [groups, logp_ranked], "labels": labels},
        shader="smoothCircle",
        colormap=[custom_cmap, "viridis"],
        point_scale=0.25,
        max_point_size=20,
        categorical=[True, False],
        has_legend=True,
        legend_labels=labels_groups,
        selected_labels=["SMILES", "Dashboard", "Name"],
        series_title=["Group", "logP"],
        max_legend_label=[None, str(round(max(logp)))],
        min_legend_label=[None, str(round(min(logp)))],
        title_index=2,
        legend_title="",
    )

    f.add_tree("dsstoxtree", {"from": s, "to": t}, point_helper="DSSTox")

    f.plot("dsstox", template="smiles")


if __name__ == "__main__":
    main()
