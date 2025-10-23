import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    Left circular analyzer (LCA)

    Polychromatic polarization assembly (PPA)

    Rotatable linear polarizer(RLP)

    Olympus U-DPTS

    Use 20x 

    1 Microscope White balance

    2. Identify zero angle for PPA
    Start from Preliminary zero position of PPA
    	 +5 , -5 curve 
      Identify RLP zero position
    Set 0 deg

     3. Align PPA
    	+/- 5deg minimum difference between angles for colorless background

    4.  Use test slide : 
    star should be white as possible at zero RLP and aligned PPA
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import smart_wsi_scanner as sws
    import numpy as np
    import matplotlib.pyplot as plt
    import pathlib
    import tifffile as tf
    return np, pathlib, plt, sws, tf


@app.cell
def _(sws):
    config = sws.ConfigManager()
    config.list_configs()
    return (config,)


@app.cell
def _():
    return


@app.cell
def _(config, pathlib):
    # MIKE'S CONFIG
    yaml_path = r"D:\2025QPSC\smartpath_configurations\config_PPM.yml"
    # Load configuration using the config manager
    ppm_settingsx = config.load_config_file(yaml_path)
    loci_rsc_file = str(
        pathlib.Path(yaml_path).parent / "resources" / "resources_LOCI.yml"
    )
    loci_resources = config.load_config_file(loci_rsc_file)
    ppm_settingsx.update(loci_resources)

    # check stage limits
    print(ppm_settingsx['stage'])
    # check for BF or PPM
    print(ppm_settingsx['ppm_optics']) #= "1")
    # check for the rotation stage loaded with config
    print(ppm_settingsx['modalities']['ppm']['rotation_stage']['device'] )#= "PIZStage")
    return (ppm_settingsx,)


@app.cell
def _(ppm_settingsx, sws):
    core,studio = sws.init_pycromanager()
    microscope_hardware = sws.PycromanagerHardware(core,studio,ppm_settingsx)
    return core, microscope_hardware


@app.cell
def _(mo):
    range_slider = mo.ui.range_slider(start = -10, stop = 10, step = 1,
        label="Select Angle Range (degrees)",show_value=True)
    range_slider
    return (range_slider,)


@app.cell
def _(range_slider):
    start_angle,stop_angle = range_slider.value
    return start_angle, stop_angle


@app.cell
def _(microscope_hardware):
    microscope_hardware.set_psg_ticks(0)
    return


@app.cell
def _(microscope_hardware):
    microscope_hardware.get_psg_ticks()
    return


@app.cell
def _():
    return


@app.cell
def _(core, microscope_hardware, plt, start_angle, stop_angle):
    arr = []
    n_images = len(range(start_angle,stop_angle))

    for ix,angle in enumerate(range (start_angle,stop_angle,1)):
        anglex = angle*1000 + 50277
        core.set_position("PIZStage",anglex)
        core.wait_for_device("PIZStage")
        img,tags = microscope_hardware.snap_image()
        plt.subplot(n_images//4,5,ix+1)
        plt.imshow(img)
        plt.title(f"Z={angle}")
        print(angle, core.get_position("PIZStage"))
        arr.append(img)
    return (arr,)


@app.cell
def _(np, plt, tf):
    plt.figure()
    arr= np.array(arr)
    plt.plot(range (-7,6,1),arr.sum(axis=(1,2,3)))
    tf.imwrite("background.tif", arr)
    return (arr,)


if __name__ == "__main__":
    app.run()
