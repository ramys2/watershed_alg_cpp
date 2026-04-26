cmake_minimum_required(VERSION 3.14)
project(watershed_alg)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

include(FetchContent)

# -------------------------------------------------------------------
# 1. External Dependencies (FetchContent)
# -------------------------------------------------------------------

# Core ImGui
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.90.4
)
FetchContent_MakeAvailable(imgui)

# SFML
FetchContent_Declare(
    sfml
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG        2.6.1
)

# ImGui-SFML
FetchContent_Declare(
    imgui-sfml
    GIT_REPOSITORY https://github.com/SFML/imgui-sfml.git
    GIT_TAG        v2.6
)

set(IMGUI_DIR ${imgui_SOURCE_DIR} CACHE PATH "Path to ImGui" FORCE)
FetchContent_MakeAvailable(sfml imgui-sfml)

# Native File Dialog Extended
FetchContent_Declare(
    nfd
    GIT_REPOSITORY https://github.com/btzy/nativefiledialog-extended.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(nfd)

# -------------------------------------------------------------------
# 2. System Packages (OpenCV + GTK)
# -------------------------------------------------------------------

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK3 REQUIRED IMPORTED_TARGET gtk+-3.0)

# -------------------------------------------------------------------
# 3. Executable Definition
# -------------------------------------------------------------------

add_executable(watershed_app
    src/Main.cpp
    src/controller/Controller.cpp
    src/model/ImageData.cpp
    src/service/image_service.cpp
    src/service/image_loader.cpp
    ${NFD_SOURCES}
)

# -------------------------------------------------------------------
# 4. Inclusion and Linking
# -------------------------------------------------------------------

# SFML and ImGui-SFML headers are managed automatically via targets
target_include_directories(watershed_app PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${NFD_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(watershed_app
    PRIVATE
    sfml-graphics
    sfml-window
    sfml-system
    ImGui-SFML::ImGui-SFML
    ${OpenCV_LIBS}
    nfd
    PkgConfig::GTK3
    GL
)
