include_directories(include)

find_package(TBB REQUIRED)

set(CMAKE_CXX_FLAGS "-march=native")

include_directories("./include")

file(GLOB_RECURSE LIB_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")

add_library(annlib STATIC ${LIB_FILES})

target_link_libraries(annlib PUBLIC TBB::tbbmalloc TBB::tbb)

add_subdirectory(test)