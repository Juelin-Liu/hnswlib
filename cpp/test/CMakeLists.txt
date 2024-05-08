find_package(GTest REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})

add_executable(distance_tests l2_test.cpp ip_test.cpp l1_test.cpp)
target_link_libraries(distance_tests PRIVATE GTest::gtest_main annlib)