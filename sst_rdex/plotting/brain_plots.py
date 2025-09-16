"""Brain visualization utilities."""

import pandas as pd
import numpy as np
import neurotools.plotting as ntp

from nilearn.datasets import fetch_atlas_surf_destrieux
from neurotools.plotting.ref import SurfRef


def broadcast_to_fsaverage(fis_agg: pd.Series, n_vertices=10242) -> pd.DataFrame:
    """Broadcast feature importance to fsaverage surface."""
    lh = np.zeros(n_vertices)
    rh = np.zeros(n_vertices)

    lh[: len(fis_agg)] = fis_agg
    rh[: len(fis_agg)] = fis_agg

    lh_df = pd.DataFrame({"data": lh, "hemi": "lh"})
    rh_df = pd.DataFrame({"data": rh, "hemi": "rh"})

    return pd.concat([lh_df, rh_df], axis=0).reset_index()


def load_destrieux_atlas():
    """Load Destrieux atlas."""
    atlas = fetch_atlas_surf_destrieux()
    return atlas


def map_destrieux(
    fis_agg: pd.Series,
    atlas=None,
    n_vertices=10242,
    include_unknown=False,
):
    """Map feature importance to Destrieux atlas regions."""
    if atlas is None:
        atlas = load_destrieux_atlas()

    lh_labels = atlas["map_left"]
    rh_labels = atlas["map_right"]

    if not include_unknown:
        labs_to_exclude = [
            "Unknown",
            "Medial_wall",
        ]
    else:
        labs_to_exclude = []

    lh_out = np.zeros(n_vertices)
    rh_out = np.zeros(n_vertices)

    for idx, lab in enumerate(atlas["labels"]):

        if lab.decode() in labs_to_exclude:
            continue

        lab_idx = np.where(lh_labels == idx)[0]
        lh_out[lab_idx] = fis_agg[lab_idx].mean()

        lab_idx = np.where(rh_labels == idx)[0]
        rh_out[lab_idx] = fis_agg[lab_idx].mean()

    lh_df = pd.DataFrame({"data": lh_out, "hemi": "lh"})
    rh_df = pd.DataFrame({"data": rh_out, "hemi": "rh"})

    return pd.concat([lh_df, rh_df], axis=0).reset_index()


def draw_plot(lh, rh, ax, mode, cmap="bwr", vmin=None, vmax=None, avg_method=None):
    """Draw brain surface plot."""
    if avg_method is None:

        def avg_method(x):
            return x[x.abs().idxmax()]

    fig_dict = {
        "lh": SurfRef.from_file("fsaverage", "lh", "inflated"),
        "rh": SurfRef.from_file("fsaverage", "rh", "inflated"),
    }

    if mode == "avg":
        ntp.plot_surf_stat_map(
            fig_dict["lh"],
            lh,
            axes=ax[0],
            view="lateral",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            avg_method=avg_method,
        )
        ntp.plot_surf_stat_map(
            fig_dict["rh"],
            rh,
            axes=ax[1],
            view="lateral",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            avg_method=avg_method,
        )
    elif mode == "agg":
        ntp.plot_surf_stat_map(
            fig_dict["lh"],
            lh,
            axes=ax[0],
            view="lateral",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ntp.plot_surf_stat_map(
            fig_dict["rh"],
            rh,
            axes=ax[1],
            view="lateral",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )


def get_global_minmax(lh, rh):
    """Get global min/max values from left and right hemisphere data."""
    all_data = pd.concat([lh, rh], axis=0)
    abs_max = all_data.abs().max()
    return -abs_max, abs_max


def make_roi_model_plot(plot_df, target, ax, vmin, vmax, cmap="bwr"):
    """Make ROI model plot."""
    import seaborn as sns

    plot_df_model = plot_df[plot_df["target"] == target]
    plot_df_model = plot_df_model.pivot_table(
        index="roi", columns=["correct", "condition"], values="fis"
    )

    sns.heatmap(
        plot_df_model,
        ax=ax,
        cmap=cmap,
        center=0,
        square=False,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"shrink": 0.8},
        xticklabels=True,
        yticklabels=True,
    )

    ax.set_title(target, fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
