from smartpath import *
from smartpath_qpscope import *
import sys
import os


sp = smartpath(core)
qp = smartpath_qpscope()

if len(sys.argv) == 5:
    self_filename, projectsFolderPath, sampleLabel, scan_type, region = sys.argv
else:
    self_filename = r"C:\Users\lociuser\Codes\smart-wsi-scanner\minimal_qupathrunner.py"
    projectsFolderPath = r"C:\Users\lociuser\Codes\MikeN\data\slides"
    sampleLabel = "First_Test3"
    scan_type = "4x_bf_1"
    region = "bounds"  # or a centroid from the qupath annotation. eg "2012-2323"
    # TODO may change to universal centroid_index naming

if core.get_property(*camm.obj_slider) == camm_.CAMM_20X_BF.objective_position_label:
    camm.imaging_mode = camm_.CAMM_20X_BF
if core.get_property(*camm.obj_slider) == camm_.CAMM_4X_BF.objective_position_label:
    camm.imaging_mode = camm_.CAMM_4X_BF
if core.get_property(*camm.obj_slider) == camm_.CAMM_20X_MPM.objective_position_label:
    camm.imaging_mode = camm_.CAMM_20X_BF


swap_lens = False
# TODO :
if swap_lens:
    if scan_type.upper().startswith("20X"):
        sp.swap_objective_lens(core, camm, camm_.CAMM_20X_BF)
        core.set_property(*camm.lamp, 4)
        print("QP: moved to 20X")
    elif scan_type.upper().startswith("4X"):
        sp.swap_objective_lens(core, camm, camm_.CAMM_4X_BF)
        print("QP: moved to 4X")
        core.set_property(*camm.lamp, 2)
    else:
        print(f"{scan_type}", file=sys.stderr)


q = qpscope_project(
    projectsFolderPath=projectsFolderPath,
    sampleLabel=sampleLabel,
    scan_type=scan_type,
    region=region,
    tile_config="TileConfiguration.txt",
)

print("QP: Input TileConfig:", q.path_tile_configuration)
positions = qp.read_TileConfiguration_coordinates(q.path_tile_configuration)
print("QP: Number of tile positions:", len(positions))
suffix_length = "06"
af_position_indices = qp.get_autofocus_positions(positions, camm, 3)

qp.scan_using_positions(
    sp,
    camm,
    save_folder=q.path_output,
    positions=positions,
    id1=q.acq_id,
    core=core,
    autofocus_indices=af_position_indices,
    suffix_length=suffix_length,
)


os.rename(
    q.path_tile_configuration, str(q.path_tile_configuration).replace(".txt", "QP.txt")
)

## writing new tileconfig with filenames = file_id / id1
new_tile_config = str(q.path_tile_configuration)
qp.write_tileconfig(
    positions=positions,
    tileconfig_path=new_tile_config,
    id1=q.acq_id,
)
## writing tileconfig for gridstitcher with 1micron pixelsize
new_tile_config = str(q.path_tile_configuration)[:-4] + "GS.txt"
qp.write_tileconfig(
    positions=[k / camm.imaging_mode.pixelsize for k in positions],
    tileconfig_path=new_tile_config,
    id1=q.acq_id,
)
