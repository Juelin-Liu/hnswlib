set(CMAKE_CXX_STANDARD 20)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

add_executable(prof_hnsw main.cc utils.cc)

target_link_libraries(prof_hnsw PRIVATE hnswlib cnpy spdlog::spdlog TBB::tbb)