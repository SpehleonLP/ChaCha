cmake_minimum_required(VERSION 3.15)
project(chacha VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB_RECURSE CHACHA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
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
