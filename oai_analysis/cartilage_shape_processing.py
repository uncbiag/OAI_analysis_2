import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def __generate_mask(vertice, mask, interx, intery, startx, starty):
    import math
    x, y = zip(*vertice)
    for i in range(len(vertice)):
        idx = math.floor((x[i] - startx) / interx)
        idy = math.floor((y[i] - starty) / intery)
        mask[idy, idx] = False
        for j in range(idx - 1, idx + 2):
            for k in range(idy - 1, idy + 2):
                mask[k, j] = False
    return mask


def __map_thickness_to_2D_projection(embedded, thickness, ninter=100, min_thickness=-1., fpth_np=None, fpth_png=None,
                                     name=''):
    """

    :param embedded: numpy array, 2d embedded atlas mesh
    :param thickness: numpy array, thickness
    :param ninter: mask resolution
    :param min_thickness: the threshold on minumum of thickness
    :param fpth_np: saving path of numpy array of 2d thickness map
    :param fpth_png: saving path of 2d project image
    :param name: projection name
    :return:
    """

    xmin, xmax, ymin, ymax = min(embedded[:, 0]), max(embedded[:, 0]), min(embedded[:, 1]), max(embedded[:, 1])
    rangex = xmax - xmin
    rangey = ymax - ymin
    interx = rangex / ninter
    intery = rangey / ninter
    xi = np.arange(xmin - interx * 5, xmax + interx * 5, interx)
    yi = np.arange(ymin - intery * 5, ymax + intery * 5, intery)
    xi, yi = np.meshgrid(xi, yi)
    mask = np.zeros_like(xi) == 0
    mask = __generate_mask(embedded, mask, interx, intery, xi.min(), yi.min())
    x, y = zip(*embedded)
    if min_thickness > 0:
        thickness[thickness < min_thickness] = min_thickness
    z = thickness[:embedded.shape[0]]  # old mapping has 4 vertices less
    zi = griddata((x, y), z, (xi, yi), method='linear')
    # zi[mask] = 100
    zi[mask] = np.nan

    # zi now has the thickness, we can write this out as a numpy file
    np.save(file=fpth_np, arr=zi)

    contour_num = 80
    maxz = max(z)
    plt.contourf(xi, yi, zi, np.arange(0.0, maxz + maxz / contour_num, maxz / contour_num))
    plt.axis('equal')
    # plt.xlabel('xi', fontsize=16)
    # plt.ylabel('yi', fontsize=16)
    plt.axis('off')
    plt.colorbar().ax.tick_params(labelsize=10)
    font = {'size': 10}
    plt.title(name, font)
    if fpth_png is not None:
        plt.savefig(fpth_png, dpi=300)
        plt.close('all')
    else:
        plt.show()
        plt.clf()


def map_thickness_to_2D_projection(atlas_mesh_with_thickness, atlas_2d_map_file=None, map_2d_base_filename=None,
                                   name='', overwrite=False):
    """
    Map thickness of a registered mesh to the 2d projection of the atlas mesh
    :param atlas_mesh_with_thickness: atlas mesh which the source mesh was registered to
    :param map_2d_base_filename: filename to save the 2d contour with mapped thickness (as png) and the raw values as a numpy array
    :param  name: name of the projection
    :param  overwrite: overwrite if the files already exist
    :return:
    """

    map_2d_file_np = map_2d_base_filename  # numpy will automatically add .npy as suffix
    map_2d_file_png = map_2d_base_filename + '.png'

    if overwrite is False:
        if os.path.exists(map_2d_file_np) and os.path.exists(map_2d_file_png):
            print(
                'Thickness 2D projection, not recomputing as {} and {} exist.'.format(map_2d_file_np, map_2d_file_png))
            return

    if type(atlas_mesh_with_thickness) == str:
        atlas_mesh_with_thickness = mp.read_vtk_mesh(atlas_mesh_with_thickness)
    elif isinstance(atlas_mesh_with_thickness, vtk.vtkPolyData):
        pass  # OK
    else:
        raise TypeError("atlas mesh is either a file path or a vtkPolyData")

    embedded = np.load(atlas_2d_map_file)
    # thickness = atlas_mesh_with_thickness.GetPointData().GetScalars()  # equivalent to below
    thickness_vtk = atlas_mesh_with_thickness.GetPointData().GetArray(0)
    thickness = vtk_to_numpy(thickness_vtk)
    print(np.where(thickness == 0)[0].shape)

    __map_thickness_to_2D_projection(embedded, thickness, ninter=300, min_thickness=-1, fpth_np=map_2d_file_np,
                                     fpth_png=map_2d_file_png, name=name)


def thickness_3d_to_2d(mapped_mesh, mesh_type: str, output_filename):
    ATLAS_DIR = pathlib.Path(__file__).parent / "data" / "atlases" / "atlas_60_LEFT_baseline_NMI"
    map_thickness_to_2D_projection(mapped_mesh, ATLAS_DIR / f"{mesh_type}_inner_embedded.npy", output_filename,
                                   name=f'{mesh_type}_2D_map', overwrite=True)
