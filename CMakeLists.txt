cmake_minimum_required(VERSION 3.14) # Verze 3.14+ je nutná pro FetchContent_MakeAvailable
project(watershed_alg)

# 1. Globální nastavení
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

include(FetchContent)

# -------------------------------------------------------------------
# 2. Externí závislosti přes FetchContent (Místo složky libs/)
# -------------------------------------------------------------------

# 1. Core ImGui (This is the missing piece)
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.90.4 # Use a version compatible with imgui-sfml 2.6
)
FetchContent_MakeAvailable(imgui)

# 2. SFML
FetchContent_Declare(
    sfml
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG        2.6.1
)

# 3. ImGui-SFML
FetchContent_Declare(
    imgui-sfml
    GIT_REPOSITORY https://github.com/SFML/imgui-sfml.git
    GIT_TAG        v2.6
)

# IMPORTANT: Tell ImGui-SFML where the core ImGui files are located
set(IMGUI_DIR ${imgui_SOURCE_DIR} CACHE PATH "Path to ImGui" FORCE)

# Now make SFML and the wrapper available
FetchContent_MakeAvailable(sfml imgui-sfml)

# -------------------------------------------------------------------
# 3. Systémové balíčky (OpenCV a GTK)
# -------------------------------------------------------------------
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK3 REQUIRED IMPORTED_TARGET gtk+-3.0)

# -------------------------------------------------------------------
# 4. Definice spustitelného souboru
# -------------------------------------------------------------------

# NFD si zatím necháme jako lokální soubory, pokud pro něj nemáš CMake balíček
set(NFD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/static/nfd")
set(NFD_SOURCES
    ${NFD_DIR}/nfd_gtk.cpp
    # ${NFD_DIR}/nfd_common.c # Odkomentuj, pokud dostaneš "undefined reference"
)

add_executable(watershed_app
    src/Main.cpp
    src/controller/Controller.cpp
    src/model/ImageData.cpp
    src/service/image_service.cpp
    src/service/image_loader.cpp
    ${NFD_SOURCES}
)

# -------------------------------------------------------------------
# 5. Inkluze a Linkování
# -------------------------------------------------------------------

# SFML a ImGui-SFML si hlavičky spravují samy přes targety (netřeba je sem psát)
target_include_directories(watershed_app PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${NFD_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(watershed_app
    PRIVATE
    # Místo libsfml-graphics.so používáme logické jméno targetu [cite: 2, 3]
    sfml-graphics
    sfml-window
    sfml-system
    ImGui-SFML::ImGui-SFML
    ${OpenCV_LIBS}
    PkgConfig::GTK3
    GL
)

# Poznámka: RPATH a link_directories pro SFML už nejsou potřeba,
# protože se SFML kompiluje přímo jako součást tvého projektu.
