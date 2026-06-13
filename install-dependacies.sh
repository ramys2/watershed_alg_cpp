#!/usr/bin/env bash
set -euo pipefail

# Install system dependencies for watershed_alg_cpp.
# Assumes Debian/Ubuntu or a compatible apt-based distro.

# Build tools:
# - cmake: required by CMakeLists.txt
# - build-essential: C++ compiler/toolchain for C++20
# - git: required because CMake FetchContent downloads dependencies from GitHub
# - pkg-config: required by find_package(PkgConfig) and GTK lookup
#
# Project libraries:
# - libopencv-dev: required by find_package(OpenCV REQUIRED)
# - libgtk-3-dev: required by pkg_check_modules(GTK3 gtk+-3.0)
# - libgl1-mesa-dev: required by target_link_libraries(... GL)
#
# SFML source-build prerequisites:
# SFML itself is fetched by CMake, but building it on Linux needs these
# development packages for graphics/window/system/audio support.
sudo apt-get update

sudo apt-get install -y \
  build-essential \
  cmake \
  git \
  pkg-config \
  libopencv-dev \
  libgtk-3-dev \
  libgl1-mesa-dev \
  libx11-dev \
  libxrandr-dev \
  libxcursor-dev \
  libudev-dev \
  libfreetype6-dev \
  libopenal-dev \
  libvorbis-dev \
  libogg-dev \
  libflac-dev

# Note:
# SFML, ImGui, ImGui-SFML, and nativefiledialog are fetched by CMake, so this
# script installs their Linux build prerequisites rather than installing those
# libraries directly.
