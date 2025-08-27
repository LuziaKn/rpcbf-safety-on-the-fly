import contourpy
import matplotlib.pyplot as plt
import numpy as np
import shapely
from loguru import logger
from matplotlib.colors import to_rgba

from og.jax_types import BBFloat

from rpcbf.plotting.poly_to_patch import poly_to_patch

def is_open(contour, bb_Xs, bb_Ys):
    # Check if any of the points are at the bounds but not closed
    # This is just an example condition, customize as needed
    # [left, bottom, right, top]
    open_idx = [np.any(contour[:, 0] == np.min(bb_Xs)), np.any(contour[:, 1] == np.min(bb_Ys)), np.any(contour[:, 0] == np.max(bb_Xs)), np.any(contour[:, 1] == np.max(bb_Ys))]
    is_open = np.sum(open_idx) > 1
    return is_open, open_idx


def add_boundary_lines(bb_Xs, bb_Ys, bb_V, contour_gen):
    contours = contour_gen.lines(0.0)  # Extract contour lines
    closed_contours = []

    C_is_open_idx = []
    for contour in contours:
        is_open_bool, is_open_idx = is_open(contour, bb_Xs, bb_Ys)
        C_is_open_idx.append(is_open_idx)
    is_open_idx_all = np.any(C_is_open_idx, axis=0)



        # if is_open_bool:  # Function to detect open contours
        #     # Add artificial lines to close the contour
        #     new_contour = close_contour_with_boundaries(contour, is_open_idx, bb_Xs, bb_Ys)
        #     # remove points that are not further away than other contour points
        #     for contour2 in contours:
        #         if contour2 is not contour:
        #             for point in new_contour:
        #                 if is_open_idx[0]:
        #                     if contour2[:,0] < point[0]:
        #     contour = np.vstack((contour, point))
        #     closed_contours.append(contour)

    return closed_contours

def close_contour_with_boundaries(contour, is_open_idx, bb_Xs, bb_Ys):
    # For instance, append points along the boundary to form a closed polygon
    new_contour = None
    if is_open_idx[0]:
        new_contour = np.vstack((new_contour, np.transpose(np.array([bb_Xs[:,0], bb_Ys[:,0]]))))
    if is_open_idx[1]:
        new_contour = np.vstack((new_contour, np.transpose(np.array([bb_Xs[0,:], bb_Ys[0,:]]))))
    if is_open_idx[2]:
        new_contour = np.vstack((new_contour,np.transpose(np.array([bb_Xs[:,-1], bb_Ys[:,-1]]))))
    if is_open_idx[3]:
        new_contour = np.vstack((new_contour, np.transpose(np.array([bb_Xs[-1,:], bb_Ys[-1,:]]))))

    return new_contour

def plot_levelset(
    bb_Xs: BBFloat,
    bb_Ys: BBFloat,
    bb_V: BBFloat,
    color,
    inner_alpha: float,
    ax: plt.Axes,
    lw: float = 1.0,
    ls: str = "-",
    merge: bool = True,
    zorder: int = 4,
    min_area_ratio: float = 0.01,
    label: str = "",
    **kwargs,
):

    contour_gen = contourpy.contour_generator(bb_Xs, bb_Ys, bb_V)
    closed_contours_lines = contour_gen.lines(0.0)  # Extract contour lines
    #closed_contours_lines = add_boundary_lines(bb_Xs, bb_Ys, bb_V, contour_gen)


    # One array for each line. (n_pts, 2)
    paths: list[np.ndarray] = closed_contours_lines
    if len(paths) > 1:
        if not merge:
            assert len(paths) == 1, "Should only have 1 path"

        # Remove all paths smaller than n points.
        areas = [shapely.Polygon(path).area for path in paths]
        max_area = max(areas)

        filtered_paths = paths
        if min_area_ratio is not None:
            filtered_paths = []
            for path, area in zip(paths, areas):
                print("Ratio: {:.3f}".format(area / max_area))
                if area / max_area >= min_area_ratio:
                    filtered_paths.append(path)

        # Merge the verts.
        logger.warning("Found {} -> {} paths! Merging...".format(len(paths), len(filtered_paths)))
        verts = np.concatenate([path for path in filtered_paths])
    else:
        verts = paths[0]

    nom_shape = shapely.Polygon(verts)
    facecolor = to_rgba(color, inner_alpha)
    nom_patch = poly_to_patch(
        nom_shape, facecolor=facecolor, edgecolor=color, lw=lw, linestyle=ls, **kwargs, zorder=zorder, label=label
    )
    ax.add_patch(nom_patch)

    return nom_patch, label