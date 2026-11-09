add_library(getline OBJECT getline.c)

# upstream tests (unchanged sources): detections and poses against the
# true values of three photos
add_executable(test_detection test_detection.c)
target_link_libraries(test_detection ${PROJECT_NAME} getline)

add_executable(test_tag_pose_estimation test_tag_pose_estimation.c)
target_link_libraries(test_tag_pose_estimation ${PROJECT_NAME} getline)

set(TEST_IMAGE_NAMES
    "33369213973_9d9bb4cc96_c"
    "34085369442_304b6bafd9_c"
    "34139872896_defdb2f8d9_c"
)

foreach(IMG IN LISTS TEST_IMAGE_NAMES)
    add_test(NAME test_detection_${IMG}
             COMMAND $<TARGET_FILE:test_detection> data/${IMG}
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_test(NAME test_tag_pose_estimation_${IMG}
             COMMAND $<TARGET_FILE:test_tag_pose_estimation>
             data/${IMG}.jpg data/${IMG}.txt
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endforeach()

# the detector with images of different sizes
add_executable(test_multiple_sizes test_multiple_sizes.c)
target_link_libraries(test_multiple_sizes ${PROJECT_NAME})
add_test(NAME test_multiple_sizes
         COMMAND $<TARGET_FILE:test_multiple_sizes> data/33369213973_9d9bb4cc96_c.jpg
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# golden test: bit-exact output (hex floats) for 18 parameter sets
if(BUILD_EXAMPLES)
    add_test(NAME test_golden
             COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_golden.sh $<TARGET_FILE:apriltag_fast_demo>
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
