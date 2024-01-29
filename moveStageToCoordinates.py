import sys

def init_pycromanager():
    from pycromanager import Core, Studio
    core = Core()
    studio = Studio()
    core.set_timeout_ms(20000)
    return core, studio

core,studio = init_pycromanager()

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
    
# Check if there are exactly two command-line arguments
if len(sys.argv) == 3: # 55.22  22.33
    x = sys.argv[1]
    y = sys.argv[2]

    if (is_float(x) and is_float(y)):        
        X = float(x)
        Y = float(y)       
        config = {
        'hard-limit-z' : [-8500.0, 17000.0],
        'hard-limit-x' : [-26000, 13000],     # X 40 cm,
        #v1 hard-limit-x=[-5000.0, 40000.0],  # X 45 cm
        'hard-limit-y' : [-17600,3500] ,      # Y 20 cm,
        #v1 hard-limit-y: [-4200.0, 25000.0], # Y 30 cm, 
        #v0 hard-limit-y: (-2200, 19000.0)    # Y 21 cm, version 0.0
        'hard-limit-f' : [-19000, 0,],
        }

        XLIMS = config['hard-limit-x']
        YLIMS = config['hard-limit-y']

        if XLIMS[0] < X < XLIMS[1]:
            pass
        else:
            print(f" X={X} Out of config[hard-limit-x] in yaml {XLIMS}", file=sys.stderr)
            sys.exit(1)
        if YLIMS[0] < Y < YLIMS[1]:
            pass    
        else:
            print(f" Y={Y} Out of config[hard-limit-y] in yaml {YLIMS}", file=sys.stderr)
            sys.exit(1)
        
        core.set_xy_position(X,Y)
        core.wait_for_device(core.get_xy_stage_device())


    else:
        print("Invalid arguments. Both X and Y must be doubles.", file=sys.stderr)
        sys.exit(1)
else:
    print("Expected two arguments, X and Y as doubles", file=sys.stderr)
    sys.exit(1)


