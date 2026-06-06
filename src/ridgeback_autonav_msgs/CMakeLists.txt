cmake_minimum_required(VERSION 3.8)
project(ridgeback_autonav_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(action_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SignDetection.msg"
  "msg/SignRegistry.msg"
  "msg/MissionStatus.msg"
  "srv/GetSigns.srv"
  "srv/StartExploration.srv"
  "action/NavigateToPose.action"
  DEPENDENCIES std_msgs geometry_msgs action_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
