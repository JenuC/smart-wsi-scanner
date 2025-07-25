# Define the strings
$x = "-1260"
$y = "5000"
$z = "-300"
$r = "85"

#XY
    #get_stageXY
    #move_stageXY -x $x -y $y
    #get_stageXY
#Z
    #get_stageZ
    #move_stageZ -z $z
    #get_stageZ

#R
    #get_stageR
    #move_stageR -angle $r
    #get_stageR

#
acquisition_workflow 

#snap 
