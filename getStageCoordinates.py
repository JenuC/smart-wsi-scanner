
def init_pycromanager():
    from pycromanager import Core, Studio
    core = Core()
    studio = Studio()
    core.set_timeout_ms(20000)
    return core, studio

## takes 1.2 seconds to run
core,studio = init_pycromanager()
xy = core.get_xy_stage_position()
print(xy.x, xy.y)
