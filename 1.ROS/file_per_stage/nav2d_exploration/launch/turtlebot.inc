define kinect ranger
(
  sensor
  (
    range_max 5.0
    fov 58.0
    samples 1280
  )
  # generic model properties
  color "black"
  size [ 0.06 0.15 0.03 ]
)

define turtlebot position
(
  pose [ 0.0 0.0 0.0 0.0 ]

  odom_error [0.0 0.0 0.0 0.0]

  size [ 0.2552 0.2552 0.40 ]
  origin [ 0.0 0.0 0.0 0.0 ]
  gui_nose 1
  drive "diff"
  color "gray"
  laser_return 0
  
  kinect(pose [ 0.0 0.0 -0.11 0.0 ]) # kinect in the middle
  
  localization "gps"
)
