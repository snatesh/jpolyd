cmake_minimum_required(VERSION 3.16)
project(jpolyd LANGUAGES C CXX)

# -------- options --------
option(JPOLYD_BUILD_CAPI "Build the C API shared library" ON)
option(JPOLYD_ENABLE_TEST "Add a python test target" ON)

# Default to Release if nothing specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Tell FindBLAS / FindLAPACK which vendor you prefer
set(BLA_VENDOR OpenBLAS)
find_package(BLAS   REQUIRED)
find_package(LAPACK REQUIRED)
find_library(LAPACKE_LIB lapacke REQUIRED)
message(STATUS "Found LAPACKE: ${LAPACKE_LIB}")


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
    c_api/src/jquad_tprod_c.cpp
  )
  target_include_directories(jpolyd_c PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/c_api/include
  )
  target_link_libraries(jpolyd_c 
    PRIVATE 
      jpolyd 
      ${LAPACKE_LIB}
      openblas
  )

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


# -------- optional python test --------
if(JPOLYD_ENABLE_TEST)
  find_package(Python COMPONENTS Interpreter REQUIRED)
  enable_testing()
  add_test(
    NAME python_jweight
    COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/testing/jweight_test.py
  )
  add_test(
    NAME jquad_tprod
    COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/testing/jquad_tprod_test.py
  )
endif()

# --- Python site selection: user (no sudo) vs system ---
option(JPOLYD_PY_INSTALL_USER "Install Python bits into user site-packages" ON)

find_package(Python COMPONENTS Interpreter REQUIRED)

# Pick target site-packages (user by default)
if(JPOLYD_PY_INSTALL_USER)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import site; print(site.getusersitepackages())"
    OUTPUT_VARIABLE PY_SITE
    OUTPUT_STRIP_TRAILING_WHITESPACE)
else()
  execute_process(  # system site (may need sudo)
    COMMAND ${Python_EXECUTABLE} -c "import site,sys; print(next((p for p in site.getsitepackages() if 'site-packages' in p), ''))"
    OUTPUT_VARIABLE PY_SITE
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

message(STATUS "Python install target: ${PY_SITE}")

# Install the Python wrapper
install(FILES ${CMAKE_SOURCE_DIR}/python/libjpolyd_loader.py DESTINATION ${PY_SITE})
install(FILES ${CMAKE_SOURCE_DIR}/python/jweight.py DESTINATION ${PY_SITE})
install(FILES ${CMAKE_SOURCE_DIR}/python/jquad_tprod.py DESTINATION ${PY_SITE})

# Install the shared lib next to it so ctypes can find it without extra paths
if(JPOLYD_BUILD_CAPI)
  install(TARGETS jpolyd_c
    LIBRARY DESTINATION ${PY_SITE}   # .so/.dylib
    RUNTIME DESTINATION ${PY_SITE}   # .dll on Windows
    ARCHIVE DESTINATION ${PY_SITE})  # (rarely used)
endif()

# -------- summary --------
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Install dir: ${JPOLYD_INSTALL_DIR}")
