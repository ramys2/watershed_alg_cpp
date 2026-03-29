#include <SFML/Graphics.hpp>
#include <SFML/Config.hpp>
#include <iostream>

// --- NEW: OpenCV Headers ---
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include "imgui.h"
#include "imgui-SFML.h"

int main() {
    std::cout << "--- Version Check ---" << std::endl;
    std::cout << "SFML: " << SFML_VERSION_MAJOR << "." << SFML_VERSION_MINOR << "." << SFML_VERSION_PATCH << std::endl;
    std::cout << "ImGui: " << IMGUI_VERSION << std::endl;
    std::cout << "OpenCV: " << CV_VERSION << std::endl;
    std::cout << "---------------------" << std::endl;

    sf::RenderWindow window(sf::VideoMode(1280, 720), "Pop_OS ImGui + SFML + OpenCV");
    window.setFramerateLimit(60);

    if (!ImGui::SFML::Init(window)) return -1;

    sf::Clock deltaClock;
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            ImGui::SFML::ProcessEvent(window, event);
            if (event.type == sf::Event::Closed) window.close();
        }

        ImGui::SFML::Update(window, deltaClock.restart());

        ImGui::Begin("Hello, Pop_OS!");
        ImGui::Text("OpenCV Version: %s", CV_VERSION); // Display version in UI
        if (ImGui::Button("Click to print OpenCV info")) {
            std::cout << "OpenCV is linked and responding!" << std::endl;
        }
        ImGui::End();

        window.clear();
        ImGui::SFML::Render(window);
        window.display();
    }

    ImGui::SFML::Shutdown();
    return 0;
}
