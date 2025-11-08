cmake_minimum_required(VERSION 3.16)
project(jpolyd LANGUAGES C CXX)

# -------- options --------
option(JPOLYD_BUILD_CAPI "Build the C API shared library" ON)
option(JPOLYD_ENABLE_TEST "Add a python test target" ON)

# Default to Release if nothing specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -------- output directories (all local) --------
set(JPOLYD_INSTALL_DIR ${CMAKE_SOURCE_DIR}/install)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# -------- header-only core --------
add_library(jpolyd INTERFACE)
target_include_directories(jpolyd INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(jpolyd INTERFACE cxx_std_17)

# -------- C API shared lib --------
if(JPOLYD_BUILD_CAPI)
  add_library(jpolyd_c SHARED
    c_api/src/jweight_c.cpp
  )
  target_include_directories(jpolyd_c PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/c_api/include
  )
  target_link_libraries(jpolyd_c PRIVATE jpolyd)

  # link libm for lgamma() if on UNIX
  if(UNIX AND NOT APPLE)
    target_link_libraries(jpolyd_c PRIVATE m)
  endif()

  set_target_properties(jpolyd_c PROPERTIES
    OUTPUT_NAME jpolyd
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
  )
endif()

# -------- local install rules --------
install(DIRECTORY include/ DESTINATION ${JPOLYD_INSTALL_DIR}/include)
install(DIRECTORY c_api/include/ DESTINATION ${JPOLYD_INSTALL_DIR}/include)

if(JPOLYD_BUILD_CAPI)
  install(TARGETS jpolyd_c
    LIBRARY DESTINATION ${JPOLYD_INSTALL_DIR}/lib
    ARCHIVE DESTINATION ${JPOLYD_INSTALL_DIR}/lib
    RUNTIME DESTINATION ${JPOLYD_INSTALL_DIR}/bin
  )
endif()

# -------- optional python test --------
if(JPOLYD_ENABLE_TEST)
  find_package(Python COMPONENTS Interpreter REQUIRED)
  enable_testing()
  add_test(
    NAME python_jweight
    COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/testing/jweight_test.py
  )
endif()

# -------- summary --------
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Install dir: ${JPOLYD_INSTALL_DIR}")
