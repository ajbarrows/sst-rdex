"""Plotting utilities for brain-behavior analysis."""

from .data_processing import (
    relabel_plotting_data,
    sort,
    absmax,
    get_fullrang_minmax,
    format_rois,
    format_for_plotting,
    gather_fis,
)

from .brain_plots import (
    broadcast_to_fsaverage,
    load_destrieux_atlas,
    map_destrieux,
    draw_plot,
    get_global_minmax,
    make_roi_model_plot,
)

from .statistical_plots import (
    make_effect_compare_plot,
    make_paper_effectsize_plot,
    make_corplot,
    make_feat_imp_radar_plot,
)

from .plot_generators import (
    make_collage_plot,
    make_colorbar,
    produce_effectsize_plot,
    produce_supplement_plots,
    produce_plots,
)

__all__ = [
    # Data processing
    "relabel_plotting_data",
    "sort",
    "absmax",
    "get_fullrang_minmax",
    "format_rois",
    "format_for_plotting",
    "gather_fis",
    # Brain plots
    "broadcast_to_fsaverage",
    "load_destrieux_atlas",
    "map_destrieux",
    "draw_plot",
    "get_global_minmax",
    "make_roi_model_plot",
    # Statistical plots
    "make_effect_compare_plot",
    "make_paper_effectsize_plot",
    "make_corplot",
    "make_feat_imp_radar_plot",
    # Plot generators
    "make_collage_plot",
    "make_colorbar",
    "produce_effectsize_plot",
    "produce_supplement_plots",
    "produce_plots",
]
