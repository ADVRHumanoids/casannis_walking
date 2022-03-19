# casannis_walking
Trajectory Optimization based on CasADi, implemented for IIT's Centauro robot.

<!--img src="https://user-images.githubusercontent.com/75118133/159123067-991317f2-fb3e-4f27-a904-88bc969467bf.gif" width="750"-->
<!--img src="https://user-images.githubusercontent.com/75118133/159121224-dac623b1-5c6d-4d6d-a026-5046bfc84920.gif" width="750"-->

<p float="left">
  <img src="https://user-images.githubusercontent.com/75118133/159123067-991317f2-fb3e-4f27-a904-88bc969467bf.gif" width="400" />
  <img width="10" />
  <img src="https://user-images.githubusercontent.com/75118133/159121224-dac623b1-5c6d-4d6d-a026-5046bfc84920.gif" width="400" /> 
</p>
 
:heavy_check_mark: Offline/online trajectory optimization for quadruped robots and quadruped manipulators carrying heavy payload.  
:heavy_check_mark: Implemented using [CasADi] and its interface to [Ipopt] solver.  
:heavy_check_mark: Based on ROS.  

<br>

## Dependencies
The repo has been tested on Ubuntu 18.04 and 20.04, using the XbotCore 2 and Cartesian Interface developed within the HHCM lab of IIT.
To use the repo you will need:
* [centauro_contact_detection]
* [iit-centauro-ros-pkg]
* [xbot2_examples]
* [centauro_cartesio]
* [cartesio_collision_support]
* [yiannis_centauro_pytools]
* [hhcm_perception] (optional)

## Install & Build
casannis_walking is a catkin package. To install it you need to `git clone` into your workspace:

`git clone https://github.com/ADVRHumanoids/casannis_walking.git`

Build with catkin_make or catkin build

  
## Run
  Launch the program using
  ```bash
  roslaunch towr_ros towr_ros.launch  # debug:=true  (to debug with gdb)
  ```
  Click in the xterm terminal and hit 'o'. 
  
  Information about how to tune the paramters can be found [here](http://docs.ros.org/api/towr/html/group__Parameters.html). 

## Publications
The repo is related with the following publication:
Dadiotis I., Laurenzi A., Tsagarakis N., “Trajectory Optimization for Quadruped Mobile Manipulators that Carry Heavy
Payload” (submitted)


## Contributors 
Ioannis Dadiotis

Arturo Laurenzi

[Ipopt]: https://github.com/coin-or/Ipopt
[CasADi]: https://web.casadi.org/
[centauro_contact_detection]: https://github.com/ADVRHumanoids/centauro_contact_detection
[iit-centauro-ros-pkg]: https://github.com/ADVRHumanoids/iit-centauro-ros-pkg
[xbot2_examples]: https://github.com/ADVRHumanoids/xbot2_examples
[centauro_cartesio]: https://github.com/ADVRHumanoids/centauro_cartesio
[cartesio_collision_support]: https://github.com/ADVRHumanoids/cartesio_collision_support
[hhcm_perception]: https://github.com/ADVRHumanoids/de_luca_perception_navigation
[yiannis_centauro_pytools]: https://github.com/ADVRHumanoids/yiannis_centauro_pytools
