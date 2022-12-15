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
:heavy_check_mark: Implemented in Python, based on ROS.  

<br>

## Dependencies
The repo has been tested on Ubuntu 18.04 and 20.04, using the XbotCore 2 and Cartesian Interface developed within the HHCM lab of IIT.
To use the repo you will need:
* [centauro_contact_detection]
* [iit-centauro-ros-pkg]
* [xbot2_examples]
* [centauro_cartesio]
* [cartesio_collision_support] (optional)
* [yiannis_centauro_pytools] (optional)
* [hhcm_perception] (optional)

## Install & Build
casannis_walking is a catkin package. To install it you need to `git clone` into your workspace:

`git clone https://github.com/ADVRHumanoids/casannis_walking.git`

Build with `catkin_make` / `catkin build`

  
## Run
The repo is **not maintained** to enable off-the-self deploying, but rather consists a record of the work done for the corresponding publication mentioned below. People interested are encouraged to contact the author.

  For offline trajectory optimization try
  ```bash
  mon launch casannis_walking cartesio.launch  
  mon launch casannis_walking gait.launch
  ```
  Or
  ```bash
  mon launch casannis_walking cartesio_with_arm_ee.launch  
  mon launch casannis_walking gait_payload.launch
  ```
  For online trajectory optimization (under development) try
  ```bash
  mon launch casannis_walking cartesio.launch  
  rosrun casannis_walking online_simple_gait_replay_node.py
  mon launch casannis_walking simple_gait_online_planner.launch
  ```

## Publications
The repo is related with the following publication:

Dadiotis I., Laurenzi A., Tsagarakis N., “Trajectory Optimization for Quadruped Mobile Manipulators that Carry Heavy
Payload”, 2022 IEEE-RAS International Conference on Humanoid Robots (Humanoids 2022), Ginowan, Okinawa, Japan

Available: https://arxiv.org/abs/2210.06803


## Author 
Ioannis Dadiotis, ioannis.dadiotis@iit.it

[Ipopt]: https://github.com/coin-or/Ipopt
[CasADi]: https://web.casadi.org/
[centauro_contact_detection]: https://github.com/ADVRHumanoids/centauro_contact_detection
[iit-centauro-ros-pkg]: https://github.com/ADVRHumanoids/iit-centauro-ros-pkg
[xbot2_examples]: https://github.com/ADVRHumanoids/xbot2_examples
[centauro_cartesio]: https://github.com/ADVRHumanoids/centauro_cartesio
[cartesio_collision_support]: https://github.com/ADVRHumanoids/cartesio_collision_support
[hhcm_perception]: https://github.com/ADVRHumanoids/de_luca_perception_navigation
[yiannis_centauro_pytools]: https://github.com/ADVRHumanoids/yiannis_centauro_pytools
