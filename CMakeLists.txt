cmake_minimum_required(VERSION 3.10)
project(watershed_alg)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK3 REQUIRED IMPORTED_TARGET gtk+-3.0)

find_package(OpenCV REQUIRED)

# 2. Define the paths to your manually added ImGui files
set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/static/imgui")
set(IMGUI_SFML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/static/imgui-sfml")
set(NFD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/static/nfd")
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(SFML_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/shared/sfml/include")
set(SFML_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/shared/sfml/lib")

# 3. Collect the ImGui source files to be compiled into your binary
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_SFML_DIR}/imgui-SFML.cpp
)

set(NFD_SOURCES
    ${NFD_DIR}/nfd_gtk.cpp
    # ${NFD_DIR}/nfd_common.c  <-- Add this if you get "undefined reference" errors later
)

# 4. Create the executable including both your code and ImGui's code
add_executable(
    watershed_app
    ${IMGUI_SOURCES}
    ${NFD_SOURCES}
    src/Main.cpp
    src/controller/Controller.cpp
    src/model/ImageData.cpp
    src/service/image_service.cpp
    src/service/image_loader.cpp
)

# 5. Tell the compiler where to find the .h headers
target_include_directories(watershed_app PUBLIC
    ${IMGUI_DIR}
    ${IMGUI_SFML_DIR}
    ${NFD_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${INCLUDE_DIR}
    ${GTK3_INCLUDE_DIRS}
    ${SFML_INCLUDE_DIR}
)

# 6. Link against system SFML and OpenGL
# ImGui requires OpenGL (GL) to render on Linux
target_link_libraries(
    watershed_app
    PRIVATE
    ${SFML_LIB_DIR}/libsfml-graphics.so
    ${SFML_LIB_DIR}/libsfml-window.so
    ${SFML_LIB_DIR}/libsfml-system.so
    GL
    ${OpenCV_LIBS}
    PkgConfig::GTK3
)

link_directories(${SFML_LIB_DIR})

set_target_properties(watershed_app PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${SFML_LIB_DIR}"
)
