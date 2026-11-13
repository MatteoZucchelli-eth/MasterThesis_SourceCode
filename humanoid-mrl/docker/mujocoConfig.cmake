# Minimal mujocoConfig.cmake with namespace target

# Root of this installation (directory containing include/, lib/, bin/)
get_filename_component(_MUJOCO_PREFIX "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)

set(mujoco_INCLUDE_DIRS "${_MUJOCO_PREFIX}/include")

set(mujoco_LIBRARIES "${_MUJOCO_PREFIX}/lib/libmujoco.so")

# Provide imported target with namespace "mujoco::mujoco"
if(NOT TARGET mujoco::mujoco)
  add_library(mujoco::mujoco SHARED IMPORTED)
  set_target_properties(
    mujoco::mujoco
    PROPERTIES IMPORTED_LOCATION "${mujoco_LIBRARIES}"
               INTERFACE_INCLUDE_DIRECTORIES "${mujoco_INCLUDE_DIRS}")
endif()
