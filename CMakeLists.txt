cmake_minimum_required(VERSION 2.6 FATAL_ERROR)


project(rotate)

 

find_package(PCL 1.5 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(BOOST_LIBS program_options serialization random system)
FIND_PACKAGE(Boost 1.48.0 COMPONENTS ${BOOST_LIBS} REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
MESSAGE("Boost information:") 
MESSAGE("  Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}") 
MESSAGE("  Boost_LIBRARIES: ${Boost_LIBRARIES}") 
MESSAGE("  Boost_LIBRARY_DIRS: ${Boost_LIBRARY_DIRS}") 

add_executable (rotate rotate.cpp) 

target_link_libraries(rotate  ${Boost_LIBRARIES} ${PCL_LIBRARIES})