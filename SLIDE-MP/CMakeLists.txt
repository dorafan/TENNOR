cmake_minimum_required(VERSION 3.16)
project(SLIDE_MP LANGUAGES C CXX)

# Target: Ubuntu 24.04 with GCC 13. SLIDE code uses the paper's -O0;
# dependencies retain their own Release configuration.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(SLIDE_SOURCES
    SLIDE/Bucket.cpp
    SLIDE/DensifiedMinhash.cpp
    SLIDE/DensifiedWtaHash.cpp
    SLIDE/Layer.cpp
    SLIDE/LSH.cpp
    SLIDE/MurmurHash.cpp
    SLIDE/Network.cpp
    SLIDE/Node.cpp
    SLIDE/srp.cpp
    SLIDE/WtaHash.cpp
)
set(SLIDE_HEADERS
    SLIDE/Bucket.h
    SLIDE/Config.h
    SLIDE/DensifiedMinhash.h
    SLIDE/DensifiedWtaHash.h
    SLIDE/Layer.h
    SLIDE/LSH.h
    SLIDE/MurmurHash.h
    SLIDE/Network.h
    SLIDE/Node.h
    SLIDE/srp.h
    SLIDE/WtaHash.h
)
# *_Origin.cpp files are historical alternatives, not additional sources.
# InputSpecificActiveNetworkFactory.h is not used by this executable.
set(missing_sources "")
foreach(source IN LISTS SLIDE_SOURCES SLIDE_HEADERS)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
        list(APPEND missing_sources "${source}")
    endif()
endforeach()
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/SLIDE/main.cpp")
    list(APPEND missing_sources "SLIDE/main.cpp")
endif()
if(missing_sources)
    list(JOIN missing_sources "\n  " missing_sources_text)
    message(FATAL_ERROR "Required source files are missing:\n  ${missing_sources_text}\nRestore these files from the matching source version before building.")
endif()

set(CNPY_GIT_COMMIT "HEAD" CACHE STRING "CNPY Git reference; optionally set an exact commit")
if(CNPY_GIT_COMMIT STREQUAL "")
    set(CNPY_GIT_COMMIT "HEAD" CACHE STRING "CNPY Git reference; optionally set an exact commit" FORCE)
endif()
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(Git REQUIRED)
include(ExternalProject)

set(dependency_prefix "${CMAKE_BINARY_DIR}/ep")
set(zlib_static "${dependency_prefix}/lib/libz.a")
set(cnpy_static "${dependency_prefix}/lib/libcnpy.a")

ExternalProject_Add(ZLIB
    PREFIX "${CMAKE_BINARY_DIR}/_deps/zlib"
    GIT_REPOSITORY https://github.com/madler/zlib.git
    GIT_TAG v1.2.11
    UPDATE_COMMAND ""
    CMAKE_ARGS
        "-DCMAKE_INSTALL_PREFIX:PATH=${dependency_prefix}"
        "-DINSTALL_BIN_DIR:PATH=${dependency_prefix}/bin"
        "-DINSTALL_INC_DIR:PATH=${dependency_prefix}/include"
        "-DINSTALL_LIB_DIR:PATH=${dependency_prefix}/lib"
        "-DINSTALL_MAN_DIR:PATH=${dependency_prefix}/share/man"
        "-DINSTALL_PKGCONFIG_DIR:PATH=${dependency_prefix}/share/pkgconfig"
        "-DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}"
        -DCMAKE_BUILD_TYPE:STRING=Release
    BUILD_BYPRODUCTS "${zlib_static}"
)

ExternalProject_Add(CNPY
    DEPENDS ZLIB
    PREFIX "${CMAKE_BINARY_DIR}/_deps/cnpy"
    GIT_REPOSITORY https://github.com/sarthakpati/cnpy.git
    GIT_TAG "${CNPY_GIT_COMMIT}"
    CMAKE_ARGS
        "-DCMAKE_INSTALL_PREFIX:PATH=${dependency_prefix}"
        -DCMAKE_INSTALL_LIBDIR:STRING=lib
        "-DZLIB_INCLUDE_DIR:PATH=${dependency_prefix}/include"
        "-DZLIB_LIBRARY:FILEPATH=${zlib_static}"
        "-DZLIB_LIBRARY_DEBUG:FILEPATH=${zlib_static}"
        "-DZLIB_LIBRARY_RELEASE:FILEPATH=${zlib_static}"
        "-DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}"
        "-DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}"
        -DBUILD_SHARED_LIBS:BOOL=OFF
        -DCMAKE_BUILD_TYPE:STRING=Release
    BUILD_BYPRODUCTS "${cnpy_static}"
)

add_library(SLIDE_LIB STATIC ${SLIDE_SOURCES} ${SLIDE_HEADERS})
add_dependencies(SLIDE_LIB CNPY)
target_include_directories(SLIDE_LIB PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/SLIDE"
    "${dependency_prefix}/include"
)
# Target options follow configuration flags, so -O0 overrides Release's -O3.
target_compile_options(SLIDE_LIB PRIVATE -O0)
target_link_libraries(SLIDE_LIB PUBLIC OpenMP::OpenMP_CXX "${cnpy_static}" "${zlib_static}")

add_executable(runme SLIDE/main.cpp)
add_dependencies(runme CNPY)
target_compile_options(runme PRIVATE -O0)
target_link_libraries(runme PRIVATE SLIDE_LIB)
set_target_properties(runme PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
message(STATUS "SLIDE-MP: -O0, C++11, OpenMP; CNPY reference ${CNPY_GIT_COMMIT}")
