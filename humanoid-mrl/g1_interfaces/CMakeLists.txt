cmake_minimum_required(VERSION 3.8)
project(g1_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find build dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Find message dependencies
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(unitree_hg REQUIRED)

# Generate custom interfaces
rosidl_generate_interfaces(
  ${PROJECT_NAME}
  "msg/ControlStep.msg"
  "msg/WallPlane.msg"
  DEPENDENCIES
  geometry_msgs
  std_msgs
  unitree_hg)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
