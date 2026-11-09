cmake_minimum_required(VERSION 3.16)
# The version is that of the upstream API this library implements.
project(apriltag VERSION 3.4.5 LANGUAGES C)

if(POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif()
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_EXAMPLES "Build example executables" ON)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "apriltags-fast supports Linux only (the worker pool uses futexes)")
endif()

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

add_compile_options(-Wall -Wextra -Wno-shift-negative-value)

file(GLOB COMMON_SRC ${CMAKE_CURRENT_SOURCE_DIR}/common/*.c)
file(GLOB TAG_FILES ${CMAKE_CURRENT_SOURCE_DIR}/tag*.c)
set(FAST_SRC fast/apriltag_all.c fast/families.c)

add_library(${PROJECT_NAME} apriltag.c apriltag_pose.c ${COMMON_SRC} ${TAG_FILES} ${FAST_SRC})
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY C_STANDARD 11)
set_property(TARGET ${PROJECT_NAME} PROPERTY C_EXTENSIONS ON)

# The fast detector: -O2 (what it is tuned for), no FP contraction (the
# output must not depend on FMA), no errno from sqrt (or it does not
# vectorize). For gcc also -fno-trapping-math (clang's default; gcc
# vectorizes trunc only with it) and the dynamic vectorizer cost model
# (gcc -O2 takes only loops without an epilogue).
set(FAST_FLAGS -ffp-contract=off -fno-math-errno $<$<NOT:$<CONFIG:Debug>>:-O2>)
if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    list(APPEND FAST_FLAGS -fno-trapping-math -fvect-cost-model=dynamic)
endif()
set_source_files_properties(fast/apriltag_all.c PROPERTIES COMPILE_OPTIONS "${FAST_FLAGS}")

if(CMAKE_C_COMPILER_ID MATCHES "Clang" AND NOT APPLE)
    target_link_options(${PROJECT_NAME} PRIVATE "-Wl,-z,relro,-z,now,-z,defs")
endif()

find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Threads::Threads m)

set_target_properties(${PROJECT_NAME} PROPERTIES SOVERSION 3 VERSION ${PROJECT_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "d")

include(GNUInstallDirs)
target_include_directories(${PROJECT_NAME} PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/apriltag>")

# install the header hierarchy as upstream does, and the fast API
file(GLOB HEADER_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h common/*.h)
list(APPEND HEADER_FILES fast/apriltag_fast.h)
foreach(HEADER ${HEADER_FILES})
    string(REGEX MATCH "(.*)[/\\]" DIR ${HEADER})
    install(FILES ${HEADER} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/${DIR})
endforeach()

# export library
set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(version_config "${generated_dir}/${PROJECT_NAME}ConfigVersion.cmake")
set(project_config "${generated_dir}/${PROJECT_NAME}Config.cmake")
set(targets_export_name "${PROJECT_NAME}Targets")
set(config_install_dir "${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}/cmake")

include(CMakePackageConfigHelpers)
configure_package_config_file(
        "CMake/apriltagConfig.cmake.in"
        "${project_config}"
        INSTALL_DESTINATION "${config_install_dir}"
)
write_basic_package_version_file("${version_config}" COMPATIBILITY SameMajorVersion)

install(TARGETS ${PROJECT_NAME} EXPORT ${targets_export_name}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
install(EXPORT ${targets_export_name}
    NAMESPACE apriltag::
    DESTINATION ${config_install_dir})
install(FILES ${project_config} ${version_config} DESTINATION ${config_install_dir})
export(TARGETS apriltag
    NAMESPACE apriltag::
    FILE ${generated_dir}/${targets_export_name}.cmake)

configure_file(${PROJECT_NAME}.pc.in ${PROJECT_NAME}.pc @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")

if(BUILD_EXAMPLES)
    # the upstream demo, unchanged
    add_executable(apriltag_demo example/apriltag_demo.c)
    target_link_libraries(apriltag_demo ${PROJECT_NAME})
    # a demo of the fast API
    add_executable(apriltag_fast_demo example/fast_demo.c)
    target_link_libraries(apriltag_fast_demo ${PROJECT_NAME})
    install(TARGETS apriltag_demo apriltag_fast_demo RUNTIME DESTINATION bin)
endif()

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(test)
endif()
