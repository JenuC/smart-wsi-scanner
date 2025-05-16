
import qp_stage as qp

print(qp.current_position_xyz)
qp.get_stageR()
qp.get_stageXY()
qp.get_stageZ()

## FIXME : need this to work for tests
# qp.move_stageZ(z=qp.current_position_xyz.z+10)

## FIXME : init for pymmcoreplus failing (need a separate init in the smartpath)

## TODO: need a logic for falling back on different hardware controllers : speed vs usage

## snap

## tile-config


