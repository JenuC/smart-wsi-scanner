from smartpath import *
from smartpath_qpscope import *

sp = smartpath(core)
qp = smartpath_qpscope()
q = qpscope_project()

if core.get_property(*camm.obj_slider) == camm_.CAMM_20X_BF.objective_position_label:
    camm.imaging = camm_.CAMM_20X_BF
if core.get_property(*camm.obj_slider) == camm_.CAMM_4X_BF.objective_position_label:
    camm.imaging = camm_.CAMM_4X_BF
if core.get_property(*camm.obj_slider) == camm_.CAMM_20X_MPM.objective_position_label:
    camm.imaging = camm_.CAMM_20X_BF


q = qpscope_project()
q.path_tile_configuration

q = qpscope_project(
    projectsFolderPath=r"C:\Users\lociuser\Codes\MikeN\data\slides",
    sampleLabel="First_Test",
    scan_type="4x_bf_1",
    region="2447_1631",
    tile_config="TileConfiguration.txt",
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
    autofocus_indices=autofocus_indices,
)
