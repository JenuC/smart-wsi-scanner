from smartpath import *
from smartpath_qpscope import *
import sys

sp = smartpath(core)
qp = smartpath_qpscope()
q = qpscope_project()

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
    camm.imaging = camm_.CAMM_20X_BF
if core.get_property(*camm.obj_slider) == camm_.CAMM_4X_BF.objective_position_label:
    camm.imaging = camm_.CAMM_4X_BF
if core.get_property(*camm.obj_slider) == camm_.CAMM_20X_MPM.objective_position_label:
    camm.imaging = camm_.CAMM_20X_BF


q = qpscope_project()
q.path_tile_configuration

q = qpscope_project(
    projectsFolderPath=projectsFolderPath,
    sampleLabel=sampleLabel,
    scan_type=scan_type,
    region=region,
    tile_config="TileConfiguration_QP.txt",
)
positions = qp.read_TileConfiguration_coordinates(q.path_tile_configuration)
af_position_indices = qp.get_autofocus_positions(positions, camm, 3)
qp.scan_using_positions(
    sp,
    camm,
    save_folder=q.path_output,
    positions=positions,
    id1=q.acq_id,
    core=core,
    autofocus_indices=af_position_indices,
)
