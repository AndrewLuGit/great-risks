cmake_minimum_required(VERSION 3.16...3.22)

include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

option(FORCE_COLORED_OUTPUT "Always produce ANSI-colored output." ON)

if(FORCE_COLORED_OUTPUT)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-fdiagnostics-color=always)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-fcolor-diagnostics)
  endif()
endif()

project(
    great-risks
    VERSION 0.1
    LANGUAGES C CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -g")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")

list(APPEND GREAT_RISKS_SOURCES
  src/great_risks/simulator.cc
  src/great_risks/greedy_agent.cc
  src/great_risks/random_agent.cc
  src/great_risks/reduced_game.cc
  src/great_risks/greedy_agent_reduced.cc
  src/great_risks/mcts_agent_reduced.cc
  src/great_risks/mcts_agent_greedy.cc
  src/great_risks/mcts_agent_random.cc
)

add_library(great_risks_lib
  ${GREAT_RISKS_SOURCES}
)

target_include_directories(great_risks_lib
  SYSTEM PUBLIC
  ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(great_risks_lib
  PRIVATE
)



add_executable(test
  scripts/test.cc
)

target_include_directories(test
    SYSTEM PRIVATE
)

target_link_libraries(test
    PRIVATE
    great_risks_lib
    nlohmann_json::nlohmann_json
)

add_executable(agent_game
  scripts/agent_game.cc
)

target_include_directories(agent_game
    SYSTEM PRIVATE
)

target_link_libraries(agent_game
    PRIVATE
    great_risks_lib
    nlohmann_json::nlohmann_json
)

add_executable(agent_game_reduced
  scripts/agent_game_reduced.cc
)

target_include_directories(agent_game_reduced
    SYSTEM PRIVATE
)

target_link_libraries(agent_game_reduced
    PRIVATE
    great_risks_lib
    nlohmann_json::nlohmann_json
)

# install(
#   TARGETS great_risks_lib
#   LIBRARY
#   DESTINATION 
# )
