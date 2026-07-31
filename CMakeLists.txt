cmake_minimum_required(VERSION 3.15)

# A plain `cmake -S . -B build` with no explicit -DCMAKE_BUILD_TYPE leaves
# CMAKE_BUILD_TYPE empty on single-config generators, which builds with no
# optimisation flags at all. Task 8's search (`select_candidate`) is
# expensive enough that this is not just slow but budget-breaking: measured
# at full model scale (88 animations x 157 frames x 49 joints), the general
# three-axis case takes 32.9s under RelWithDebInfo but 120.0s unoptimised --
# landing exactly on Task 16's 120s guard on this machine, and over it on a
# slower CI box. Default to RelWithDebInfo (matches what every measurement
# in this codebase's comments and reports assumes) rather than leaving the
# choice to whatever the ambient environment happens to default to; a
# caller can still override with -DCMAKE_BUILD_TYPE=... as usual.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

project(chacha VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Pick up newly added sources without a manual cmake re-run.
file(GLOB_RECURSE CHACHA_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE CHACHA_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")

add_library(chacha STATIC ${CHACHA_SOURCES} ${CHACHA_HEADERS})

# -Wall -Wextra on the library itself, not just on downstream consumers
# (tonton-example only turned these on for chacha-example). Catching real
# defects here is cheaper than catching them at the call site.
target_compile_options(chacha PRIVATE -Wall -Wextra)

# Off by default: sanitizers weren't wired into either CMake or CI before
# this option existed, which is how a live out-of-bounds read sat
# undetected until a reviewer happened to build with ASan by hand. Opt in
# with -DCHACHA_ENABLE_ASAN=ON for local debugging or a dedicated CI job;
# leave off for normal builds since it changes ABI/perf characteristics.
option(CHACHA_ENABLE_ASAN "Build chacha (and its tests) with -fsanitize=address,undefined" OFF)
if(CHACHA_ENABLE_ASAN)
    target_compile_options(chacha PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(chacha PUBLIC -fsanitize=address,undefined)
endif()

target_include_directories(chacha
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

find_package(glm QUIET)
if(glm_FOUND)
    target_link_libraries(chacha PUBLIC glm::glm)
else()
    message(STATUS "GLM not found via find_package, assuming it's in include path")
endif()

set_target_properties(chacha PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/chacha.h;${CMAKE_CURRENT_SOURCE_DIR}/include/chacha_types.h;${CMAKE_CURRENT_SOURCE_DIR}/include/chacha_stage.h;${CMAKE_CURRENT_SOURCE_DIR}/include/chacha_clean.h;${CMAKE_CURRENT_SOURCE_DIR}/include/chacha_naming.h"
)

option(CHACHA_BUILD_TESTS "Build ChaCha unit tests" ON)

if(CHACHA_BUILD_TESTS)
    find_package(GTest QUIET)
    if(GTest_FOUND)
        enable_testing()
        file(GLOB CHACHA_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
        add_executable(chacha-tests ${CHACHA_TEST_SOURCES})
        target_include_directories(chacha-tests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(chacha-tests PRIVATE chacha GTest::gtest GTest::gtest_main)
        if(glm_FOUND)
            target_link_libraries(chacha-tests PRIVATE glm::glm)
        endif()
        include(GoogleTest)
        gtest_discover_tests(chacha-tests)
    else()
        message(STATUS "GTest not found; ChaCha unit tests disabled")
    endif()
endif()
