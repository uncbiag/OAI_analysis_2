import os
import itk
import vtk
import shutil
import torch
import numpy as np
import torch.nn.functional as F
from scipy.optimize import curve_fit
from icon_registration import config
from icon_registration.losses import to_floats
from icon_registration.itk_wrapper import create_itk_transform

from oai_analysis_2.mesh_processing import get_mesh, split_mesh, get_cylinder, get_projection_from_circle_and_vertice

image_type = itk.Image[itk.D, 3]  # pixel type, dimension


def np2itk(np_image, ref_image):
    image = itk.GetImageFromArray(np_image)
    image.SetOrigin(ref_image.GetOrigin())
    image.SetDirection(ref_image.GetDirection())
    image.SetSpacing(ref_image.GetSpacing())
    return image

############ read and preprocess T2 ############
def read_t2(image_path, ref_path, side, smap_folder, create_mask=True):
    smaps = []
    volumes = 7
    image_mask = None

    # 0. get ref image info
    ref_image = itk.imread(ref_path)
    ref_spacing = ref_image.GetSpacing()
    ref_origin = ref_image.GetOrigin()
    ref_direction = ref_image.GetDirection()

    # copy s1~s7 image to the temp/ folder
    if os.path.exists(smap_folder):
        shutil.rmtree(smap_folder)
    os.makedirs(smap_folder)
    length = len([name for name in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, name))])
    for i in range(volumes):
        si_map_folder = os.path.join(smap_folder, 's'+str(i+1)+'_map')
        os.makedirs(si_map_folder)
        for j in range(i * length // volumes, (i + 1) * length // volumes):
            src = os.path.join(image_path, str(j + 1).zfill(3))
            dst = os.path.join(si_map_folder, str(j + 1).zfill(3))
            shutil.copyfile(src, dst)

    # read s1~s7 image & preprocess
    for i in range(volumes):
        ### read image ###
        si_map_folder = os.path.join(smap_folder, 's' + str(i + 1) + '_map')
        name_generator = itk.GDCMSeriesFileNames.New()
        name_generator.SetDirectory(si_map_folder)
        series_identifier = name_generator.GetSeriesUIDs()[0]
        file_names = name_generator.GetFileNames(series_identifier)

        reader = itk.ImageSeriesReader[image_type].New()
        reader.SetImageIO(itk.GDCMImageIO.New())
        reader.SetFileNames(file_names)
        reader.ForceOrthogonalDirectionOff()
        si_image = reader.GetOutput()
        si_image.Update()

        ### preprocess image ###
        # 1. set image origin, direction
        si_image.SetOrigin(ref_origin)
        si_image.SetDirection(ref_direction)

        # 2. set image spacing to the same as dess image spacing
        si_spacing = si_image.GetSpacing()
        si_size = itk.size(si_image)
        si_dimension = si_image.GetImageDimension()  # 3 for mri
        new_size = itk.Size[si_dimension]()
        for j in range(si_dimension):
            new_size[j] = int(si_spacing[j] * si_size[j] / ref_spacing[j])
        interpolator = itk.NearestNeighborInterpolateImageFunction.New(si_image)
        si_image = itk.resample_image_filter(
            si_image,
            interpolator=interpolator,
            size=new_size,
            output_spacing=ref_spacing,
            output_direction=si_image.GetDirection(),
            output_origin=si_image.GetOrigin(),
        )

        # 3. flip left knee to right knee
        if side == 'RIGHT':
            flipFilter = itk.FlipImageFilter[image_type].New()
            flipFilter.SetInput(si_image)
            flipAxes = (False, False, True)
            flipFilter.SetFlipAxes(flipAxes)
            flipFilter.Update()
            si_image = flipFilter.GetOutput()

        # 4. normalize only s1 image (for registration)
        if i == 0:
            np_si_image = itk.GetArrayViewFromImage(si_image)
            min_value = np.percentile(np_si_image, 0.1)
            max_value = np.percentile(np_si_image, 99.9)

            rescaler = itk.IntensityWindowingImageFilter[image_type, image_type].New()
            rescaler.SetInput(si_image)
            rescaler.SetWindowMinimum(min_value)
            rescaler.SetWindowMaximum(max_value)
            rescaler.SetOutputMinimum(0.)
            rescaler.SetOutputMaximum(1.)
            rescaler.Update()
            si_image = rescaler.GetOutput()

        # 5. pad & crop image size
        ref_size = itk.size(ref_image)
        si_size = itk.size(si_image)
        size_diff = ref_size - si_size
        size_diff = np.array([size_diff[i] for i in range(len(size_diff))])
        pad_size = np.where(size_diff < 0, 0, size_diff)
        lower_pad = [int(pad_size[i] / 2) if pad_size[i] % 2 == 0 else int((pad_size[i] - 1) / 2) for i in range(len(pad_size))]
        upper_pad = [int(pad_size[i] / 2) if pad_size[i] % 2 == 0 else int((pad_size[i] + 1) / 2) for i in range(len(pad_size))]

        pad_filter = itk.ConstantPadImageFilter[image_type, image_type].New()
        pad_filter.SetInput(si_image)
        pad_filter.SetPadLowerBound(lower_pad)
        pad_filter.SetPadUpperBound(upper_pad)
        pad_filter.SetConstant(0)
        pad_filter.Update()
        si_image = pad_filter.GetOutput()

        crop_size = np.where(size_diff > 0, 0, -size_diff)
        lower_crop = [int(crop_size[i] / 2) if crop_size[i] % 2 == 0 else int((crop_size[i] - 1) / 2) for i in range(len(crop_size))]
        upper_crop = [int(crop_size[i] / 2) if crop_size[i] % 2 == 0 else int((crop_size[i] + 1) / 2) for i in range(len(crop_size))]

        crop_filter = itk.CropImageFilter[image_type, image_type].New()
        crop_filter.SetInput(si_image)
        crop_filter.SetLowerBoundaryCropSize(lower_crop)
        crop_filter.SetUpperBoundaryCropSize(upper_crop)
        crop_filter.Update()
        si_image = crop_filter.GetOutput()

        smaps.append(si_image)
    # return smaps

        if i == 0 and create_mask:
            image_mask = image_type.New()
            region = itk.ImageRegion[3]()
            region.SetSize(si_size)
            region.SetIndex(itk.Index[3]([0, 0, 0]))

            image_mask.SetRegions(region)
            image_mask.Allocate()
            image_mask.FillBuffer(itk.NumericTraits[itk.D].OneValue())

            pad_filter = itk.ConstantPadImageFilter[image_type, image_type].New()
            pad_filter.SetInput(image_mask)
            pad_filter.SetPadLowerBound(lower_pad)
            pad_filter.SetPadUpperBound(upper_pad)
            pad_filter.SetConstant(0)
            pad_filter.Update()
            image_mask = pad_filter.GetOutput()

            crop_filter = itk.CropImageFilter[image_type, image_type].New()
            crop_filter.SetInput(image_mask)
            crop_filter.SetLowerBoundaryCropSize(lower_crop)
            crop_filter.SetUpperBoundaryCropSize(upper_crop)
            crop_filter.Update()
            image_mask = crop_filter.GetOutput()

    return smaps, image_mask


############ calculate T2 values ############
def linear_fit(smaps, time_period, t2_max):
    print("calculate t2 using linear least square fit")
    image_shape = smaps[0].shape
    # b=A*x
    A = []
    for i in range(len(smaps)):
        A.append([1, -time_period * (i + 2)])  # -20 ~ -70

    b = []
    for i in range(len(smaps)):
        b.append(np.log(smaps[i].flatten()))

    x = np.linalg.inv(np.matmul(np.transpose(A), A))
    x = np.matmul(x, np.transpose(A))
    x = np.matmul(x, b)

    t2 = np.nan_to_num(x[1])
    t2 = 1. / t2
    t2 = t2.reshape(image_shape)
    t2 = np.where(t2 > t2_max, t2_max, t2)
    t2 = np.where(t2 < 0, 0, t2)

    s0 = np.nan_to_num(x[0])
    s0 = np.exp(s0.reshape(image_shape))
    s0_max = np.percentile(s0, 95)
    s0 = np.where(s0 > s0_max, s0_max, s0)
    s0 = np.where(s0 < 0, 0, s0)
    return t2, s0


def nonlinear_fit(smaps, init_values, segmentation, time_period, t2_max):
    def func(x, t2, s0):
        return s0 * np.exp(-x / t2)

    print("calculate t2 using scipy curve_fit")
    position = np.argwhere(segmentation > 0.5)  # only calculate t2 inside the cartilage
    init_t2, init_s0 = init_values

    all_t2 = []
    t2 = np.zeros_like(init_t2)
    s0 = np.zeros_like(init_s0)

    x_data = []
    for i in range(len(smaps)):
        t = time_period * (i + 2)
        x_data.append(t)
    x_data = np.array(x_data)

    for i in range(len(position)):
        pos = position[i]
        y_data = [smaps[j][pos[0]][pos[1]][pos[2]] for j in range(len(smaps))]
        p0 = [init_t2[pos[0]][pos[1]][pos[2]], init_s0[pos[0]][pos[1]][pos[2]]]
        try:
            popt, _ = curve_fit(func, x_data, y_data, p0, maxfev=1000)
            t2_value, s0_value = popt
        except RuntimeError:
            t2_value = init_t2[pos[0]][pos[1]][pos[2]]
            s0_value = init_s0[pos[0]][pos[1]][pos[2]]

        t2_value = max(0, min(t2_value, t2_max))
        all_t2.append(t2_value)

        t2[pos[0]][pos[1]][pos[2]] = t2_value
        s0[pos[0]][pos[1]][pos[2]] = 0 if s0_value < 0 else s0_value

    print("median t2: ", np.median(all_t2))
    print("mean t2: ", np.mean(all_t2))
    return t2, s0


def calculate_t2(smaps, probmap):
    time_period = 10
    t2_max = 200
    t2, s0 = linear_fit(smaps, time_period, t2_max)
    t2, s0 = nonlinear_fit(smaps, [t2, s0], probmap, time_period, t2_max)
    return t2, s0


############ extract mesh ############
def get_t2_statistics(mesh, t2_image, coord_pairs, num_points=10, cropped_value=200, measures=['mean', 'median']):
    coordinates = [coord_pairs[i][0] for i in range(len(coord_pairs))]
    closest_coordinates = [coord_pairs[i][1] for i in range(len(coord_pairs))]
    coordinates = [[i / j for i, j in zip(coordinates[k], list(t2_image.GetSpacing()))] for k in range(len(coordinates))]
    closest_coordinates = [[i / j for i, j in zip(closest_coordinates[k], list(t2_image.GetSpacing()))] for k in range(len(closest_coordinates))]

    point_array = [vtk.vtkDoubleArray() for _ in range(len(measures))]
    for i in range(len(measures)):
        point_array[i].SetName(measures[i])
        point_array[i].SetNumberOfComponents(1)
        point_array[i].SetNumberOfTuples(len(coordinates))

    for i in range(len(coordinates)):
        coord = coordinates[i]
        closest_coord = closest_coordinates[i]
        all_points, all_t2 = [], []     # along the thickness line
        for j in range(num_points):
            point = [round(m + j * (n - m) / (num_points - 1)) for m, n in zip(coord, closest_coord)]
            # if list(point) in all_points:
            #     continue
            all_points.append(list(point))
            all_t2.append(t2_image.GetPixel(point))
        all_t2 = np.where(np.array(all_t2) > cropped_value, cropped_value, all_t2)
        for j in range(len(measures)):
            if measures[j] in ['mean', 'median', 'std', 'mad']:
                value = getattr(np, measures[j])(all_t2)
            elif measures[j] == 'quantile':
                value = np.quantile(all_t2, 0.25)
            elif measures[j] == 'third_quantile':
                value = np.quantile(all_t2, 0.75)
            else:
                print(measures[j], ' is not defined.')
                break
            point_array[j].SetValue(i, value)

    for i in range(len(measures)):
        mesh.GetPointData().AddArray(point_array[i])
    mesh.GetPointData().SetActiveScalars('t2_statistics')
    return mesh


def get_matching_coord(inner_mesh, outer_mesh):
    coord_pairs = []
    distance_filter = vtk.vtkImplicitPolyDataDistance()
    distance_filter.SetInput(outer_mesh)
    for i in range(inner_mesh.GetNumberOfPoints()):
        pt2 = [0, 0, 0]
        pt1 = list(inner_mesh.GetPoint(i))
        distance_filter.EvaluateFunctionAndGetClosestPoint(pt1, pt2)
        coord_pairs.append([pt1, pt2])
    return coord_pairs


def get_t2_mesh(itk_image, probmap_image, mesh_type, num_iterations=150, measures=['mean', 'median']):
    # Get mesh from itk image
    mesh = get_mesh(probmap_image, num_iterations=num_iterations)

    # Split the mesh into inner and outer
    inner_mesh, outer_mesh = split_mesh(mesh, mesh_type)

    # Get the coordinate between inner and outer mesh
    coord_pairs = get_matching_coord(inner_mesh, outer_mesh)

    t2_statistics = get_t2_statistics(inner_mesh, itk_image, coord_pairs, measures=measures)
    return t2_statistics


# almost the same as project thickness,
# except mapped_mesh.GetPointData().GetScalars() -> mapped_mesh.GetPointData().GetArray(i)
def project_statistics(mapped_mesh, mesh_type="FC", embedded=None, measure_num=0):
    def do_linear_pca(vertice, dim=3.0):
        from sklearn.decomposition import KernelPCA

        kpca = KernelPCA(n_components=2, degree=dim, n_jobs=None)
        embedded = kpca.fit_transform(vertice)
        return embedded

    def rotate_embedded(embedded, angle):
        theta = (angle / 180.0) * np.pi
        rotMatrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        embedded = c = np.dot(embedded, rotMatrix)
        return embedded

    point_data = np.array(mapped_mesh.GetPointData().GetArray(measure_num))

    if mesh_type == "FC":
        vertices = np.array(mapped_mesh.GetPoints().GetData())
        vertices[:, [1, 0]] = vertices[:, [0, 1]]
        circle, z_range = get_cylinder(vertices)
        embedded, plot_xy = get_projection_from_circle_and_vertice(vertices, circle)

        return embedded[:, 0], embedded[:, 1], point_data
    else:
        vertice = np.array(mapped_mesh.GetPoints().GetData())

        vertice_left = vertice[vertice[:, 2] < 50]
        index_left = np.where(vertice[:, 2] < 50)[0]

        vertice_right = vertice[vertice[:, 2] >= 50]
        index_right = np.where(vertice[:, 2] >= 50)[0]

        embedded_left = do_linear_pca(vertice_left)
        embedded_right = do_linear_pca(vertice_right)

        embedded_left = rotate_embedded(embedded_left, -50)
        embedded_right = rotate_embedded(embedded_right, -160)

        embedded_right[:, 0] = -embedded_right[:, 0]  # flip x

        combined_embedded_x = np.concatenate(
            [embedded_right[:, 0], embedded_left[:, 0]]
        )
        combined_embedded_y = np.concatenate(
            [embedded_right[:, 1] + 50, embedded_left[:, 1]]
        )
        combined_point_data = np.concatenate(
            [point_data[index_right], point_data[index_left]]
        )

        return combined_embedded_x, combined_embedded_y, combined_point_data


# same as functions in itk_warper, but add mask
def finetune_execute(model, image_A, image_B, image_A_mask, image_B_mask, steps):
    state_dict = model.state_dict()
    best_state_dict = model.state_dict()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00002)
    best = 100
    for _ in range(steps):
        optimizer.zero_grad()
        loss_tuple = model(image_A, image_B, image_A_mask, image_B_mask)
        print(loss_tuple)
        is_best = loss_tuple[0] < best
        if is_best:
            best_state_dict = model.state_dict()
            best = loss_tuple[0]
        loss_tuple[0].backward()
        optimizer.step()
    with torch.no_grad():
        model.load_state_dict(best_state_dict)
        loss = model(image_A, image_B, image_A_mask, image_B_mask)
    model.load_state_dict(state_dict)
    return loss


def register_pair_with_mask(
    model, image_A, image_B, image_A_mask=None, image_B_mask=None, finetune_steps=None, return_artifacts=False
) -> "(itk.CompositeTransform, itk.CompositeTransform)":

    assert isinstance(image_A, itk.Image)
    assert isinstance(image_B, itk.Image)

    # send model to cpu or gpu depending on config- auto detects capability
    model.to(config.device)

    A_npy = np.array(image_A)
    B_npy = np.array(image_B)

    assert(np.max(A_npy) != np.min(A_npy))
    assert(np.max(B_npy) != np.min(B_npy))
    # turn images into torch Tensors: add feature and batch dimensions (each of length 1)
    A_trch = torch.Tensor(A_npy).to(config.device)[None, None]
    B_trch = torch.Tensor(B_npy).to(config.device)[None, None]

    shape = model.identity_map.shape

    # Here we resize the input images to the shape expected by the neural network. This affects the
    # pixel stride as well as the magnitude of the displacement vectors of the resulting
    # displacement field, which create_itk_transform will have to compensate for.
    A_resized = F.interpolate(
        A_trch, size=shape[2:], mode="trilinear", align_corners=False
    )
    B_resized = F.interpolate(
        B_trch, size=shape[2:], mode="trilinear", align_corners=False
    )
    if image_A_mask is None:
        A_mask_resized = None
    else:
        A_mask_npy = np.array(image_A_mask)
        A_mask_trch = torch.Tensor(A_mask_npy).to(config.device)[None, None]
        A_mask_resized = F.interpolate(
            A_mask_trch, size=shape[2:], mode="trilinear", align_corners=False
        )
    if image_B_mask is None:
        B_mask_resized = None
    else:
        B_mask_npy = np.array(image_B_mask)
        B_mask_trch = torch.Tensor(B_mask_npy).to(config.device)[None, None]
        B_mask_resized = F.interpolate(
            B_mask_trch, size=shape[2:], mode="trilinear", align_corners=False
        )
    if finetune_steps == 0:
        raise Exception("To indicate no finetune_steps, pass finetune_steps=None")

    if finetune_steps == None:
        model.eval()
        with torch.no_grad():
            loss = model(A_resized, B_resized)
    else:
        model.train()
        loss = finetune_execute(model, A_resized, B_resized, A_mask_resized, B_mask_resized, finetune_steps)

    # phi_AB and phi_BA are [1, 3, H, W, D] pytorch tensors representing the forward and backward
    # maps computed by the model
    if hasattr(model, "prepare_for_viz"):
        with torch.no_grad():
            model.prepare_for_viz(A_resized, B_resized)
    phi_AB = model.phi_AB(model.identity_map)
    phi_BA = model.phi_BA(model.identity_map)

    # the parameters ident, image_A, and image_B are used for their metadata
    itk_transforms = (
        create_itk_transform(phi_AB, model.identity_map, image_A, image_B),
        create_itk_transform(phi_BA, model.identity_map, image_B, image_A),
    )
    if not return_artifacts:
        return itk_transforms
    else:
        return itk_transforms + (to_floats(loss),)