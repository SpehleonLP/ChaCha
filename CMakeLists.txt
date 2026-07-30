cmake_minimum_required(VERSION 3.15)
project(chacha VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Pick up newly added sources without a manual cmake re-run.
file(GLOB_RECURSE CHACHA_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE CHACHA_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")

add_library(chacha STATIC ${CHACHA_SOURCES} ${CHACHA_HEADERS})

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
    PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/chacha.h;${CMAKE_CURRENT_SOURCE_DIR}/include/chacha_types.h;${CMAKE_CURRENT_SOURCE_DIR}/include/chacha_stage.h;${CMAKE_CURRENT_SOURCE_DIR}/include/chacha_clean.h"
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
