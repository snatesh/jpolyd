cmake_minimum_required(VERSION 3.16)
project(jpolyd LANGUAGES C CXX)

# -------- options --------
option(JPOLYD_BUILD_CAPI "Build the C API shared library" ON)
option(JPOLYD_ENABLE_TEST "Add a python test target" ON)

# Default to Release if nothing specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Iinclude -O3 -march=native")
elseif(debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Iinclude -g -DDEBUG -march=native")
endif()



# find required packages
set(BLA_VENDOR OpenBLAS)
find_package(BLAS   REQUIRED)
find_package(LAPACK REQUIRED)
find_library(LAPACKE_LIB lapacke REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
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

# -------- test: DCT vs Jacobi block transform --------
add_executable(test_dct_vs_jacobi
  ${CMAKE_SOURCE_DIR}/src/test_dct_vs_jacobi.cpp
)

target_include_directories(test_dct_vs_jacobi PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(test_pixel_moment_ls
    src/test_pixel_moment_ls.cpp
)

add_executable(test_pixel_gram_schmidt
    src/test_pixel_gram_schmidt.cpp
)

add_executable(precompute_jacobi_Q_triangles
    src/precompute_jacobi_Q_triangles.cpp
)

add_executable(precompute_jacobi_Q_tree
    src/precompute_jacobi_Q_tree.cpp
)

add_executable(test_jacobi_block_codec
    src/test_jacobi_block_codec.cpp
)
add_executable(test_jacobi_block_quant
    src/test_jacobi_block_quant.cpp
)

add_executable(brick_tet_kernel 
    src/brick_tet_kernel.cpp
)
add_executable(brick_tet_kernel_vid 
    src/brick_tet_kernel_vid.cpp
)
add_executable(brick_tet_codec 
    src/brick_tet_codec.cpp
)
add_executable(tet_video_codec_full 
    src/tet_video_codec_full.cpp
)
add_executable(generate_glass_bowl_mesh
  src/generate_glass_bowl_mesh.cpp)
add_executable(assemble_cn_step 
  src/assemble_cn_step.cpp)

target_include_directories(assemble_cn_step PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  /usr/include/suitesparse
)
target_link_libraries(assemble_cn_step PRIVATE
  jpolyd
  OpenMP::OpenMP_CXX
  ${BLAS_LIBRARIES}
  ${LAPACK_LIBRARIES}
  ${LAPACKE_LIB}
  openblas #/usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so
  spqr
  cholmod
  umfpack
  amd
  colamd
  suitesparseconfig
  blas
  lapack
)


target_link_libraries(tet_video_codec_full PRIVATE
  jpolyd                        # your header-only target
  OpenMP::OpenMP_CXX            # needed by basis code
  ${BLAS_LIBRARIES}
  ${LAPACK_LIBRARIES}
  ${LAPACKE_LIB}
  ${OpenCV_LIBS}
  openblas #/usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so
)

target_link_libraries(brick_tet_codec PRIVATE
  jpolyd                        # your header-only target
  OpenMP::OpenMP_CXX            # needed by basis code
  ${BLAS_LIBRARIES}
  ${LAPACK_LIBRARIES}
  ${LAPACKE_LIB}
  ${OpenCV_LIBS}
  openblas #/usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so
)
target_link_libraries(brick_tet_kernel_vid PRIVATE
  jpolyd                        # your header-only target
  OpenMP::OpenMP_CXX            # needed by basis code
  ${BLAS_LIBRARIES}
  ${LAPACK_LIBRARIES}
  ${LAPACKE_LIB}
  ${OpenCV_LIBS}
  openblas #/usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so
)
target_link_libraries(brick_tet_kernel PRIVATE
  jpolyd                        # your header-only target
  OpenMP::OpenMP_CXX            # needed by basis code
  ${BLAS_LIBRARIES}
  ${LAPACK_LIBRARIES}
  ${LAPACKE_LIB}
  openblas #/usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so
)



target_link_libraries(test_jacobi_block_quant PRIVATE
  jpolyd                        # your header-only target
  OpenMP::OpenMP_CXX            # needed by basis code
  ${BLAS_LIBRARIES}
  ${LAPACK_LIBRARIES}
  ${LAPACKE_LIB}
  openblas #/usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so
)
target_link_libraries(test_jacobi_block_codec PRIVATE
  jpolyd                        # your header-only target
  OpenMP::OpenMP_CXX            # needed by basis code
  ${BLAS_LIBRARIES}
  ${LAPACK_LIBRARIES}
  ${LAPACKE_LIB}
  openblas #/usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so
)
target_link_libraries(precompute_jacobi_Q_triangles PRIVATE
  jpolyd                        # your header-only target
  OpenMP::OpenMP_CXX            # needed by basis code
  ${BLAS_LIBRARIES}
  ${LAPACK_LIBRARIES}
  ${LAPACKE_LIB}
  openblas #/usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so
)
target_link_libraries(precompute_jacobi_Q_tree PRIVATE
  jpolyd                        # your header-only target
  OpenMP::OpenMP_CXX            # needed by basis code
  ${BLAS_LIBRARIES}
  ${LAPACK_LIBRARIES}
  ${LAPACKE_LIB}
  openblas #/usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so
)
target_link_libraries(test_pixel_gram_schmidt PRIVATE
  jpolyd                        # your header-only target
  OpenMP::OpenMP_CXX            # needed by basis code
  ${BLAS_LIBRARIES}
  ${LAPACK_LIBRARIES}
  ${LAPACKE_LIB}
  openblas #/usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so
)

target_link_libraries(test_pixel_moment_ls PRIVATE
  jpolyd                        # your header-only target
  OpenMP::OpenMP_CXX            # needed by basis code
  ${BLAS_LIBRARIES}
  ${LAPACK_LIBRARIES}
  ${LAPACKE_LIB}
  openblas #/usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so
)

target_link_libraries(test_dct_vs_jacobi PRIVATE
  jpolyd                        # your header-only target
  OpenMP::OpenMP_CXX            # needed by basis code
  ${BLAS_LIBRARIES}
  ${LAPACK_LIBRARIES}
  ${LAPACKE_LIB}
  openblas #/usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so
)

set_target_properties(test_dct_vs_jacobi PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# -------- C API shared lib --------
if(JPOLYD_BUILD_CAPI)
  add_library(jpolyd_c SHARED
    c_api/src/jweight_c.cpp
    c_api/src/jquad_tprod_c.cpp
    c_api/src/jbasis_c.cpp
    c_api/src/jmat_c.cpp
    c_api/src/jevd_c.cpp
    c_api/src/thread_control.cpp
    c_api/src/jquad_optim_c.cpp
    c_api/src/jkmat.cpp
    c_api/src/jdmat.cpp
    c_api/src/jmult_c.cpp
  )
  target_include_directories(jpolyd_c PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/c_api/include
  )
  target_link_libraries(jpolyd_c 
    PRIVATE 
      jpolyd 
      OpenMP::OpenMP_CXX
      /usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so
      ${LAPACKE_LIB}
      nlopt
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
    NAME python_jquad_tprod
    COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/testing/jquad_tprod_test.py
  )
  add_test(
    NAME python_jbasis_lebesgue
    COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/testing/jbasis_test_lebesgue.py
  )
  add_test(
    NAME python_jbasis_jacobi
    COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/testing/jbasis_test_jacobi.py
  )
  add_test(
    NAME python_jbasis_quad
    COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/testing/jbasis_quad_test.py
  )
  add_test(
    NAME python_jmat
    COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/testing/jmat_test.py
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
install(FILES ${CMAKE_SOURCE_DIR}/python/jbasis.py DESTINATION ${PY_SITE})
install(FILES ${CMAKE_SOURCE_DIR}/python/jmat.py DESTINATION ${PY_SITE})
install(FILES ${CMAKE_SOURCE_DIR}/python/jevd.py DESTINATION ${PY_SITE})
install(FILES ${CMAKE_SOURCE_DIR}/python/thread_control.py DESTINATION ${PY_SITE})
install(FILES ${CMAKE_SOURCE_DIR}/python/jquad_optim.py DESTINATION ${PY_SITE})
install(FILES ${CMAKE_SOURCE_DIR}/python/jkmat.py DESTINATION ${PY_SITE})
install(FILES ${CMAKE_SOURCE_DIR}/python/jdmat.py DESTINATION ${PY_SITE})
install(FILES ${CMAKE_SOURCE_DIR}/python/jmult.py DESTINATION ${PY_SITE})

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
