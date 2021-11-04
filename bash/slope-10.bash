rosparam load $(rospack find yiannis_centauro_pytools)/yaml/slope_homing_rod_plates.yaml cartesian/home
mon launch yiannis_centauro_pytools initialize_cartesio.launch imu_available:=false
mon launch casannis_walking gait_payload.launch slope_x:=-10 target_dx:="[0.2, 0.2, 0.2, 0.2]" min_force:=150

