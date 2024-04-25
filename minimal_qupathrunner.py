from pycromanager import Core, Studio
from smartpath_libraries.sp_acquisition import SPAcquisition
import yaml
from skimage import io, img_as_uint, transform
import os
import numpy as np
import glob
import shutil
import sys
import re
from IPython.utils import io as ipio

# import pathlib

# TODO change log entirely to json scheme within folder
# TODO change debug mode printing to logging (not sure std pipe can handle log)
debug_mode = 1  # 0 OFF 1 WARN 2 DEBUG

# print(sys.path)
cwd = os.getcwd()
os.chdir(os.path.dirname(os.path.realpath(__file__)))


def read_TileConfiguration_coordinates(tile_config_path) -> list:
    coordinates = []
    with open(tile_config_path, "r") as file:
        for line in file:
            # Extract coordinates using regular expression
            match = re.search(r"\((-?\d+\.\d+), (-?\d+\.\d+)\)", line)
            if match:
                x, y = map(float, match.groups())
                coordinates.append([x, y])
    return np.array(coordinates)


def simulate_4_tiles(core, delta_xy=500.0):
    xy = core.get_xy_stage_position()
    X = xy.x
    Y = xy.y

    coordinates = np.array(
        [[X, Y], [X + delta_xy, Y], [X + delta_xy, Y + delta_xy], [X, Y + delta_xy]]
    )
    # coordinates = np.array(coordinates)
    return coordinates


if len(sys.argv) == 5:
    if debug_mode > 0:
        print("QuPath: Arguments passed, using them", sys.argv)
    self_filename, projectsFolderPath, sampleLabel, scan_type, region = sys.argv
else:
    if debug_mode > 0:
        print("QuPath: Incorrect number of arguments using default values")
    self_filename = r"C:\Users\lociuser\Codes\smart-wsi-scanner\minimal_qupathrunner.py"
    projectsFolderPath = r"C:\Users\lociuser\Codes\MikeN\data\slides"
    sampleLabel = "First_Test3"
    scan_type = "4x_bf_1"
    region = "bounds"  # or a centroid from the qupath annotation. eg "2012-2323"
    # TODO may change to universal centroid_index naming


def init_pycromanager():
    core = Core()
    studio = Studio()
    core.set_timeout_ms(20000)
    print("QuPath: Pycromanager loaded successfully")
    return core, studio


# Config Loading
config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
config = {**config["user_config"], **config["model_config"], **config["hard_config"]}
lsm_pixel_size_base = config["pixel-size-shg-base"]
bf_4x_pixel_size_base = config["pixel-size-bf-4x"]
bf_20x_pixel_size_base = config["pixel-size-bf-20x"]
camera_resolution_base = config["camera-resolution"]
if debug_mode > 1:
    print("QuPath: Loaded Config File")


brightfield_4x_background_fname = (
    "data/presets/BG_4x.tiff"  # give a default 4x background image
)
brightfield_20x_background_fname = (
    "data/presets/BG_20x.tiff"  # give a default 20x background image
)

core, studio = init_pycromanager()
if debug_mode > 1:
    print("QuPath: Pycromanager Initialized")

if core.is_sequence_running():
    studio.live().set_live_mode(False)

qupath_project_folder = os.path.join(projectsFolderPath, sampleLabel)
path_to_TileConfiguration = os.path.join(
    qupath_project_folder, scan_type, region, "TileConfiguration.txt"
)
if os.path.exists(path_to_TileConfiguration):
    coordinates = read_TileConfiguration_coordinates(path_to_TileConfiguration)
    if debug_mode > 0:
        print("QuPath: Read TileConfiguration Coordinates")
else:
    if debug_mode > 0:
        print("QuPath: TileConfiguration file not found, simulating 4 tiles")
    coordinates = simulate_4_tiles(core, delta_xy=500.0)

save_path = os.path.join(qupath_project_folder, scan_type)  # "data/acquisition"
# if not os.path.exists(save_path):
#    os.makedir(save_path)

sp_acq = SPAcquisition(
    config=config,
    mmcore=core,
    mmstudio=studio,
    bf_4x_bg=io.imread(brightfield_4x_background_fname),
    bf_20x_bg=io.imread(brightfield_20x_background_fname),
)
if debug_mode > 0:
    print("QuPath: Loaded settings to MM2")

config["Z-stage-4x"] = 0.0
config["hard-limit-x"] = [0, 40000.0]
config["hard-limit-y"] = [0, 30000.0]

acq_name = sampleLabel + "-" + scan_type


for x0, y0 in coordinates:
    if config["hard-limit-x"][0] < x0 < config["hard-limit-x"][1]:
        coordinates_within_limits_x = True
    else:
        print(f"{x0=} is out of range")
        coordinates_within_limits_x = False
        break

    if config["hard-limit-y"][0] < y0 < config["hard-limit-y"][1]:
        coordinates_within_limits_y = True
    else:
        print(f"{y0=} is out of range")
        coordinates_within_limits_y = False

coordinates_within_limits = coordinates_within_limits_y and coordinates_within_limits_x

if debug_mode > 0:
    print(f"QuPath: Coordinates within limits {coordinates_within_limits}")

# fov_x = config["pixel-size-bf-4x"] * 1392
# fov_y = config["pixel-size-bf-4x"] * 1040
# coordinates[:,1] -= fov_y
# coordinates[:,0] += fov_x/4

sp_acq.config["autofocus-speed"] = (
    4  # default is 4 ## `1-6`, the larger the faster, but potentially worse autofocus results.
)


core.set_auto_shutter(False)
core.set_shutter_open(True)

if debug_mode > 0:
    print("QuPath: Starting Acquisition")

# Need better error handling here, crashes with Nonetype

if coordinates_within_limits:
    print(f"QuPath: Before modality check for {acq_name}")
    if "-4x_bf" in acq_name.lower():
        print("QuPath: Begin 4x acquisition")
        results_4x = sp_acq.whole_slide_bf_scan(
            save_path,
            acq_name,
            coordinates,
            mag="4x",
            focus_dive=True,
            estimate_background=False,
        )
        core.set_auto_shutter(True)
        acq_id = len(glob.glob(os.path.join(save_path, acq_name + "*")))
        acq_path = os.path.join(save_path, acq_name + "_{}".format(acq_id))
        if debug_mode > 0:
            print(f"QuPath: Saved files to {acq_path}")

        position_list = coordinates
        if position_list.ndim == 3:
            p1, p2, p3 = position_list.shape
            position_list = np.reshape(position_list, [p1 * p2, p3])

        pixel_size = config["pixel-size-bf-4x"]

        # mage_list = glob.glob(os.path.join(acq_path, "*.tif"))
        image_list = sorted(
            glob.glob(os.path.join(acq_path, "*.tif")),
            key=lambda x: int(os.path.basename(x).split("-")[0]),
        )
        stitchfolder_path = os.path.join(acq_path, "stitch")
        if not os.path.exists(stitchfolder_path):
            os.mkdir(stitchfolder_path)

        #    assert len(position_list) == len( image_list ), "Number of images does not match number of positions"

        with open(
            os.path.join(stitchfolder_path, "TileConfiguration_python.txt"), "w"
        ) as text_file:
            print("dim = {}".format(2), file=text_file)
            for pos in range(position_list.shape[0]):
                x = float(
                    position_list[pos][0]
                )  # int(position_list[pos][0] / pixel_size)
                y = float(
                    position_list[pos][1]
                )  # int(position_list[pos][1] / pixel_size)
                print("{}.tif; ; ({}, {})".format(pos, x, y), file=text_file)

        if debug_mode > 0:
            print(f"QuPath: Saved new TileConfiguration.txt to {stitchfolder_path}")
            print(f"QuPath: tif-files in acq folder :{len(image_list)}")
            if debug_mode > 1:
                print(f"QuPath: Image List     :{image_list}   ")
                # print(f"QuPath: Position List  :{position_list}")

        for pos in range(len(image_list)):

            fn = image_list[pos]
            img = io.imread(fn)

            correction = False
            rotate = False
            flip_y = True
            flip_x = False

            # if correction is True and background_image is not None:
            #    img = white_balance(img, background_image)
            #    img = flat_field(img, bg_img)

            if rotate is not None:
                img = transform.rotate(img, rotate)

            if flip_y:
                img = img[::-1, :]

            if flip_x:
                img = img[:, ::-1]

            if debug_mode > 1:
                print(f"QuPath: Moving {pos}.tif to {stitchfolder_path}")

            save_filename = os.path.join(stitchfolder_path, f"{pos}.tif")

            # TODO: replace scikit iosave with tifffile with metadata
            io.imsave(
                save_filename,
                img_as_uint(img),
                check_contrast=False,
            )
            if debug_mode > 1:
                print(f"QuPath: Saved: {save_filename}")

        qupath_stitching_folder = os.path.join(
            projectsFolderPath, sampleLabel, scan_type, region
        )
        if debug_mode > 0:
            print("QuPath: Stripping Metadata For stitching ")
            print(
                f"QuPath: copying from \n{stitchfolder_path} \t to \n{qupath_stitching_folder}"
            )

        shutil.copytree(stitchfolder_path, qupath_stitching_folder, dirs_exist_ok=True)

        shutil.rmtree(stitchfolder_path)
        if debug_mode > 0:
            print(f"QuPath: Finished saving tiles for stitching at {stitchfolder_path}")
        os.chdir(cwd)

    ##TODO ok this could almost certainly be done more cleanly than giant ifs, I only changed like 2 things
    elif "-20x_bf" in acq_name.lower():
        print(f"QuPath: beginning {acq_name} acquisition")
        # results_20x = sp_acq.whole_slide_bf_scan(
        #     save_path,
        #     acq_name,
        #     coordinates,
        #     mag="20x",
        #     focus_dive=True,
        #     estimate_background=False,
        # )
        # core.set_auto_shutter(True)
        # acq_id = len(glob.glob(os.path.join(save_path, acq_name + "*")))
        # acq_path = os.path.join(save_path, acq_name + "_{}".format(acq_id))
        # if debug_mode > 0:
        #     print(f"QuPath: Saved files to {acq_path}")

        # position_list = coordinates
        # if position_list.ndim == 3:
        #     p1, p2, p3 = position_list.shape
        #     position_list = np.reshape(position_list, [p1 * p2, p3])

        # pixel_size = config["pixel-size-bf-20x"]

        # # mage_list = glob.glob(os.path.join(acq_path, "*.tif"))
        # image_list = sorted(
        #     glob.glob(os.path.join(acq_path, "*.tif")),
        #     key=lambda x: int(os.path.basename(x).split("-")[0]),
        # )
        # stitchfolder_path = os.path.join(acq_path, "stitch")
        # if not os.path.exists(stitchfolder_path):
        #     os.mkdir(stitchfolder_path)

        # #    assert len(position_list) == len( image_list ), "Number of images does not match number of positions"

        # with open(
        #     os.path.join(stitchfolder_path, "TileConfiguration_python.txt"), "w"
        # ) as text_file:
        #     print("dim = {}".format(2), file=text_file)
        #     for pos in range(position_list.shape[0]):
        #         x = float(
        #             position_list[pos][0]
        #         )  # int(position_list[pos][0] / pixel_size)
        #         y = float(
        #             position_list[pos][1]
        #         )  # int(position_list[pos][1] / pixel_size)
        #         print("{}.tif; ; ({}, {})".format(pos, x, y), file=text_file)

        # if debug_mode > 0:
        #     print(f"QuPath: Saved new TileConfiguration.txt to {stitchfolder_path}")
        #     print(f"QuPath: tif-files in acq folder :{len(image_list)}")
        #     if debug_mode > 1:
        #         print(f"QuPath: Image List     :{image_list}   ")
        #         # print(f"QuPath: Position List  :{position_list}")

        # for pos in range(len(image_list)):

        #     fn = image_list[pos]
        #     img = io.imread(fn)

        #     correction = False
        #     rotate = False
        #     flip_y = True
        #     flip_x = False

        #     # if correction is True and background_image is not None:
        #     #    img = white_balance(img, background_image)
        #     #    img = flat_field(img, bg_img)

        #     if rotate is not None:
        #         img = transform.rotate(img, rotate)

        #     if flip_y:
        #         img = img[::-1, :]

        #     if flip_x:
        #         img = img[:, ::-1]

        #     if debug_mode > 1:
        #         print(f"QuPath: Moving {pos}.tif to {stitchfolder_path}")

        #     save_filename = os.path.join(stitchfolder_path, f"{pos}.tif")

        #     # TODO: replace scikit iosave with tifffile with metadata
        #     io.imsave(
        #         save_filename,
        #         img_as_uint(img),
        #         check_contrast=False,
        #     )
        #     if debug_mode > 1:
        #         print(f"QuPath: Saved: {save_filename}")

        # qupath_stitching_folder = os.path.join(
        #     projectsFolderPath, sampleLabel, scan_type, region
        # )
        # if debug_mode > 0:
        #     print("QuPath: Stripping Metadata For stitching ")
        #     print(
        #         f"QuPath: copying from \n{stitchfolder_path} \t to \n{qupath_stitching_folder}"
        #     )

        # shutil.copytree(stitchfolder_path, qupath_stitching_folder, dirs_exist_ok=True)

        # shutil.rmtree(stitchfolder_path)
        # if debug_mode > 0:
        #     print(f"QuPath: Finished saving tiles for stitching at {stitchfolder_path}")
        # os.chdir(cwd)

        ### Other option but this looks like it involves some other functionality like stitching?

        # with open(path.join(save_path, acq_name_4x + ".pkl"), "rb") as f:
        #     loaded_results_4x = pickle.load(f)
        # sp_acq.position_list_4x = loaded_results_4x["Position list"]
        # sp_acq.z_list_4x = loaded_results_4x["Z positions"]
        # position_list_xyz = np.concatenate(
        #     (
        #         sp_acq.position_list_4x,
        #         sp_acq.z_list_4x.reshape(
        #             (
        #                 sp_acq.position_list_4x.shape[0],
        #                 sp_acq.position_list_4x.shape[1],
        #                 1,
        #             )
        #         ),
        #     ),
        #     2,
        # )
        # position_lists_20x, annotation_names = sp_acq.annotations_positionlist(
        #     image_name=acq_name_4x, out_mag="20x"
        # )
        # position_lists_20x, annotation_names = zip(
        #     *sorted(
        #         zip(position_lists_20x, annotation_names),
        #         key=lambda x: int(x[1].split("-")[-1]),
        #     )
        # )
        # ### perform the scan
        # sp_acq.position_list_20x = []
        # sp_acq.z_list_20x = []
        # for idx, (roi_pos, roi_name) in tqdm(
        #     enumerate(zip(position_lists_20x, annotation_names))
        # ):
        #     current_acq_name = acq_name + "-20x-" + annotation_names[idx]
        #     sampled_pos_xyz = sp_acq.resample_z_pos(
        #         mag="20x", xy_pos=roi_pos, xyz_pos_list_4x=position_list_xyz
        #     )
        #     results_20x = sp_acq.whole_slide_bf_scan(
        #         save_path,
        #         current_acq_name,
        #         sampled_pos_xyz,
        #         mag="20x",
        #         focus_dive=True,
        #         estimate_background=False,
        #     )
        #     sp_acq.position_list_20x.append(sampled_pos_xyz)
        #     sp_acq.z_list_20x.append(results_20x["Z positions"])
        #     sp_acq.config["Z-stage-20x"] = np.mean(np.vstack(sp_acq.z_list_20x))
        #     ### save background image and z positions, and position_list
        #     position_list = sp_acq.position_list_4x.reshape(
        #         sp_acq.position_list_4x.shape[0] * sp_acq.position_list_4x.shape[1], -1
        #     )
        #     with ipio.capture_output() as captured:
        #         sp_sti.stitch_bf(
        #             current_acq_name,
        #             mag="20x",
        #             position_list=sampled_pos_xyz,
        #             flip_y=True,
        #             correction=False,
        #             background_image=None,
        #         )
        #     sp_sti.convert_slide(mag="20x")
        #     # TODO: failing when multiple ROI
        #     try:
        #         sp_sti.clean_folders(current_acq_name)  # optional
        #     except Exception as e:
        #         print("Failed to delete temporary stitching files")
        #         print(e)
        # np.save(
        #     os.path.join(save_path, acq_name + "-20x" + "-z_pos.npy"),
        #     sp_acq.z_list_20x,
        #     allow_pickle=True,
        # )

core._close()
studio._close()
del studio
del core
if debug_mode > 0:
    print("QuPath: Pycromanager Acquisition Task Completed")
