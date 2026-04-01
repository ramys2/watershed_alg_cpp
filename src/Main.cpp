#include <SFML/Graphics.hpp>
#include <SFML/Config.hpp>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include "imgui.h"
#include "imgui-SFML.h"
#include "controller/Controller.hpp"

int main()
{
    std::cout << "--- Version Check ---" << std::endl;
    std::cout << "SFML: " << SFML_VERSION_MAJOR << "." << SFML_VERSION_MINOR << "." << SFML_VERSION_PATCH << std::endl;
    std::cout << "ImGui: " << IMGUI_VERSION << std::endl;
    std::cout << "OpenCV: " << CV_VERSION << std::endl;
    std::cout << "---------------------" << std::endl;

    sf::RenderWindow window(sf::VideoMode(1280, 720), "Pop_OS ImGui + SFML + OpenCV");
    window.setFramerateLimit(60);

    if (!ImGui::SFML::Init(window))
        return -1;

    sf::Clock deltaClock;
    Controller controller;
    while (window.isOpen())
    {
        sf::Event event;
        sf::Vector2u windowSize = window.getSize();
        while (window.pollEvent(event))
        {
            ImGui::SFML::ProcessEvent(window, event);
            if (event.type == sf::Event::Closed)
                window.close();
        }

        ImGui::SFML::Update(window, deltaClock.restart());

        controller.renderGuiElements(windowSize);
        controller.renderOriginalImage(windowSize);
        controller.renderSegmentedlImage(windowSize);

        controller.update();

        window.clear();
        ImGui::SFML::Render(window);
        window.display();
    }

    ImGui::SFML::Shutdown();
    return 0;
}
