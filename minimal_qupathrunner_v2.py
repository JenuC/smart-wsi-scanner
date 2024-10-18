from smartpath import *
from smartpath_config import *
from smartpath_qpscope import *
import sys

if len(sys.argv) == 5:
    self_filename, projectsFolderPath, sampleLabel, scan_type, region = sys.argv
else:
    self_filename = r"C:\Users\lociuser\Codes\smart-wsi-scanner\minimal_qupathrunner.py"
    projectsFolderPath = r"C:\Users\lociuser\Codes\MikeN\data\slides"
    sampleLabel = "First_Test3"
    scan_type = "4x_bf_1"
    region = "bounds"  # or a centroid from the qupath annotation. eg "2012-2323"
    # TODO may change to universal centroid_index naming

sp = smartpath(core)
qp = smartpath_qpscope()    


yaml_data = read_yaml_file(r'./smartpath_configurations/config_CAMM.yml')
camm_ = yaml_to_dataclass(yaml_data)
yaml_data = read_yaml_file(r'./smartpath_configurations/resources_LOCI.yml')
loci_ = yaml_to_dataclass(yaml_data)

camm = sp_camm_settings(stage = camm_.stage)


if core.get_property(*camm_.objectiveSlider) == camm_.imagingMode.BF_20X.objectivePositionLabel:
    camm.imaging_mode = camm_.imagingMode.BF_20X
if core.get_property(*camm_.objectiveSlider) == camm_.imagingMode.BF_4X.objectivePositionLabel:
    camm.imaging_mode = camm_.imagingMode.BF_4X


q = qpscope_project(
    projectsFolderPath=projectsFolderPath,
    sampleLabel=sampleLabel,
    scan_type=scan_type,
    region=region,
    tile_config="TileConfiguration.txt",
)

print(q.path_tile_configuration)
positions = qp.read_TileConfiguration_coordinates(q.path_tile_configuration)
print(len(positions))
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

## overwrite current tileconfig file
# new_tile_config = str(q.path_tile_configuration)[:-4] + "2.txt"
new_tile_config = q.path_tile_configuration


qp.write_tileconfig(
    positions=positions,
    tileconfig_path=new_tile_config,
    id1=q.acq_id,
)
