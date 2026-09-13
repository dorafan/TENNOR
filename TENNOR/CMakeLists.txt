cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(HashDeepLearning)


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------

# ---------------------------
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" ARCH_LOWER)
set(SLIDE_X86 FALSE)
if(ARCH_LOWER MATCHES "x86_64" OR ARCH_LOWER MATCHES "amd64" OR ARCH_LOWER MATCHES "i[3-6]86")
  set(SLIDE_X86 TRUE)
endif()
message(STATUS "Detected processor: ${CMAKE_SYSTEM_PROCESSOR} (SLIDE_X86=${SLIDE_X86})")


option(USE_SNOOPY_AVX2 "Enable AVX2 optimizations for obl_primitives on x86" ON)

if(USE_SNOOPY_AVX2 AND SLIDE_X86)
  message(STATUS "Enabling OHT_SNOOPY_AVX2 and -mavx2 -mfma")
  add_definitions(-DOHT_SNOOPY_AVX2)

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
else()
  message(STATUS "AVX2 disabled for obl_primitives (either non-x86 or USE_SNOOPY_AVX2=OFF)")
endif()

# ---------------------------

# ---------------------------
if(APPLE)
  if ("${OpenMP_LIBRARIES}" STREQUAL "")
    find_library(OpenMP_LIBRARY
      NAMES omp
    )
    find_path(OpenMP_INCLUDE_DIR
      omp.h
    )
    mark_as_advanced(OpenMP_LIBRARY OpenMP_INCLUDE_DIR)
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(OpenMP DEFAULT_MSG
      OpenMP_LIBRARY OpenMP_INCLUDE_DIR)
    if (OpenMP_FOUND)
      set(OpenMP_LIBRARIES ${OpenMP_LIBRARY} CACHE STRING "Cache OpenMP Lib" FORCE)
      set(OpenMP_INCLUDE_DIRS ${OpenMP_INCLUDE_DIR} CACHE STRING "Cache OpenMP Include" FORCE)
      set(OpenMP_COMPILE_OPTIONS -XPREPROCESSOR -fopenmp -lomp)
      add_library(OpenMP::OpenMP SHARED IMPORTED)
      set_target_properties(OpenMP::OpenMP PROPERTIES
        IMPORTED_LOCATION ${OpenMP_LIBRARIES}
        INTERFACE_INCLUDE_DIRECTORIES "${OpenMP_INCLUDE_DIRS}"
        INTERFACE_COMPILE_OPTIONS "${OpenMP_COMPILE_OPTIONS}"
      )
      add_definitions(-D_OPENMP="${OpenMP_CXX_VERSION}")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
  endif()
else()
  find_package(OpenMP REQUIRED)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# ---------------------------

# ---------------------------
include(ExternalProject)

find_package(Git REQUIRED)

option(USE_GIT_PROTOCOL "If behind a firewall turn this off to use https instead." OFF)


function(functionInstallExternalCMakeProject ep_name)
  ExternalProject_Get_Property(${ep_name} binary_dir)
  install(SCRIPT ${binary_dir}/cmake_install.cmake)
endfunction()

# ---------- ZLIB ----------
ExternalProject_Add(
  ZLIB
  DEPENDS ""
  GIT_REPOSITORY https://github.com/madler/zlib.git
  GIT_TAG v1.2.11
  SOURCE_DIR ZLIB-source
  BINARY_DIR ZLIB-build
  UPDATE_COMMAND ""
  PATCH_COMMAND ""
  # INSTALL_COMMAND ""
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX:STRING=${PROJECT_BINARY_DIR}/ep
    -DINSTALL_BIN_DIR:STRING=${PROJECT_BINARY_DIR}/ep/bin
    -DINSTALL_INC_DIR:STRING=${PROJECT_BINARY_DIR}/ep/include
    -DINSTALL_LIB_DIR:STRING=${PROJECT_BINARY_DIR}/ep/lib
    -DINSTALL_MAN_DIR:STRING=${PROJECT_BINARY_DIR}/ep/share/man
    -DINSTALL_PKGCONFIG_DIR:STRING=${PROJECT_BINARY_DIR}/ep/share/pkgconfig
    -DCMAKE_BUILD_TYPE:STRING=Release
)
functionInstallExternalCMakeProject(ZLIB)


if(WIN32)
  set(ZLIB_LIB_DEBUG   ${PROJECT_BINARY_DIR}/ep/lib/zlibstaticd.lib)
  set(ZLIB_LIB_RELEASE ${PROJECT_BINARY_DIR}/ep/lib/zlibstatic.lib)
else()
  set(ZLIB_LIB_DEBUG   ${PROJECT_BINARY_DIR}/ep/lib/libz.a)
  set(ZLIB_LIB_RELEASE ${PROJECT_BINARY_DIR}/ep/lib/libz.a)
endif()

# ---------- CNPY ----------
ExternalProject_Add(
  CNPY
  DEPENDS ZLIB
  GIT_REPOSITORY https://github.com/sarthakpati/cnpy.git
  # GIT_TAG v1.2.11
  SOURCE_DIR CNPY-source
  BINARY_DIR CNPY-build
  UPDATE_COMMAND ""
  PATCH_COMMAND ""
  # INSTALL_COMMAND ""
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    -DZLIB_INCLUDE_DIR:STRING=${PROJECT_BINARY_DIR}/ep/include
    -DZLIB_LIBRARY_DEBUG:STRING=${ZLIB_LIB_DEBUG}
    -DZLIB_LIBRARY_RELEASE:STRING=${ZLIB_LIB_RELEASE}
    -DCMAKE_INSTALL_PREFIX:STRING=${PROJECT_BINARY_DIR}/ep
    -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS}
    -DCMAKE_BUILD_TYPE:STRING=Release
)
functionInstallExternalCMakeProject(CNPY)

include_directories(${PROJECT_BINARY_DIR}/ep/include)


if(WIN32)
  set(CNPY_LIB ${PROJECT_BINARY_DIR}/ep/lib/cnpy.lib)
else()
  set(CNPY_LIB ${PROJECT_BINARY_DIR}/ep/lib/libcnpy.a)
endif()

# ---------------------------

# ---------------------------
file(GLOB_RECURSE SLIDE_SOURCES "${PROJECT_SOURCE_DIR}/SLIDE/*.cpp")
file(GLOB_RECURSE SLIDE_HEADERS "${PROJECT_SOURCE_DIR}/SLIDE/*.h")


add_library(SLIDE_LIB ${SLIDE_HEADERS} ${SLIDE_SOURCES})
add_dependencies(SLIDE_LIB CNPY)
target_link_libraries(SLIDE_LIB ${CNPY_LIB})


set(SLIDE_EXE_NAME runme)
add_executable(${SLIDE_EXE_NAME} ${PROJECT_SOURCE_DIR}/SLIDE/main.cpp)
add_dependencies(${SLIDE_EXE_NAME} SLIDE_LIB)
target_link_libraries(
  ${SLIDE_EXE_NAME}
  SLIDE_LIB
  ${CNPY_LIB}
  ${ZLIB_LIB_RELEASE}
)

install(TARGETS ${SLIDE_EXE_NAME} DESTINATION bin)
