cmake_minimum_required(VERSION 3.10)
project(watershed_alg)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# 1. Find the SFML installed on your Pop!_OS system
# This looks for the shared libraries in /usr/lib/x86_64-linux-gnu/
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

find_package(OpenCV REQUIRED)

# 2. Define the paths to your manually added ImGui files
set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/imgui")
set(IMGUI_SFML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/imgui-sfml")
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# 3. Collect the ImGui source files to be compiled into your binary
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_SFML_DIR}/imgui-SFML.cpp
)

# 4. Create the executable including both your code and ImGui's code
add_executable(
    watershed_app
    ${IMGUI_SOURCES}
    src/Main.cpp
    src/controller/Controller.cpp
    src/model/AppData.cpp
    src/service/ImageService.cpp
    src/service/ImageLoader.cpp
)

# 5. Tell the compiler where to find the .h headers
target_include_directories(watershed_app PUBLIC
    ${IMGUI_DIR}
    ${IMGUI_SFML_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${INCLUDE_DIR}
)

# 6. Link against system SFML and OpenGL
# ImGui requires OpenGL (GL) to render on Linux
target_link_libraries(
    watershed_app
    PRIVATE
    sfml-graphics
    sfml-window
    sfml-system
    GL
    ${OpenCV_LIBS}
)
