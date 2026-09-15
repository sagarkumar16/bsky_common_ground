import matplotlib as mpl
import seaborn as sns


# color styling
def set_colors():
    global complexity_cmap
    global palette

    complexity_cmap = sns.color_palette("flare_r", as_cmap=True)
    palette = sns.color_palette("YlGnBu")
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=palette)


def set_fonts(extra_params={}):
    params = {
        "font.family": "sans-serif",
        # "mathtext.fontset": "cm",
        "legend.fontsize": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "figure.titlesize": 14,
    }
    for key, value in extra_params.items():
        params[key] = value
    mpl.rcParams.update(params)
