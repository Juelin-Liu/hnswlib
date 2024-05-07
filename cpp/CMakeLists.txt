include_directories(include)

# set(CMAKE_CXX_FLAGS  "-O3 -DNDEBUG -std=c++20 -march=native -fpic -ftree-vectorize")
# set(CMAKE_CXX_FLAGS "-march=native")

include_directories("./include")

file(GLOB_RECURSE LIB_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")

add_library(annlib SHARED ${LIB_FILES})

target_compile_options(annlib PRIVATE "-march=native")

add_executable(ann ann.cpp)

target_link_libraries(ann PRIVATE annlib)