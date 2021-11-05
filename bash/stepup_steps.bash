mon launch casannis_walking gait_payload.launch min_force:=175 target_dx:="[0.3, 0.3, 0.3, 0.3]" target_dz:="[0.3,0.3,0.0,0.0]" swing_id:="[1,2,3,4]" clearance:=0.1
mon launch casannis_walking gait_payload.launch min_force:=175 target_dx:="[0.25, 0.25, 0.23, 0.23]" target_dz:="[0.0,0.0,0.0,0.0]" swing_id:="[1,2,3,4]" clearance:=0.1
mon launch casannis_walking gait_payload.launch min_force:=175 target_dx:="[0.25, 0.25, 0.22, 0.22]" target_dz:="[0.0,0.0,0.0,0.0]" swing_id:="[1,2,3,4]" clearance:=0.1
mon launch casannis_walking gait_payload.launch min_force:=250 target_dx:="[0.25, 0.25, 0.25, 0.25]" target_dz:="[0.3,0.3,0.0,0.0]" swing_id:="[3,4]" clearance:=0.1 plots:=true publish_until:=4.0

