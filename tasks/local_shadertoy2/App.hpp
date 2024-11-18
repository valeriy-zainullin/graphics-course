#pragma once

#include <etna/Window.hpp>
#include <etna/PerFrameCmdMgr.hpp>
#include <etna/ComputePipeline.hpp>
#include <etna/GraphicsPipeline.hpp>
#include <etna/Image.hpp>
#include <etna/Sampler.hpp>

#include "wsi/OsWindowingManager.hpp"


class App
{
public:
  App();
  ~App();

  void run();

private:
  void initSkyboxPipeline();
  void initMainItemPipeline();


  void drawFrame();

  void prepareSkybox(vk::CommandBuffer& cmdBuffer);
  void prepareMainItem(vk::Image& backbuffer, vk::ImageView& backbufferView, vk::CommandBuffer& cmdBuffer);

private:
  OsWindowingManager windowing;
  std::unique_ptr<OsWindow> osWindow;

  glm::uvec2 resolution;
  bool useVsync;

  std::unique_ptr<etna::Window> vkWindow;
  std::unique_ptr<etna::PerFrameCmdMgr> commandManager;

  // Добавленные поля.
  // etna::ComputePipeline  pipeline; // Remain of compute shader.
  // etna::Image            tmp_image; // Remain of compute shader, should be named tmpImage.
  etna::Sampler          sampler;
  etna::GraphicsPipeline skyboxPipeline;
  etna::GraphicsPipeline mainItemPipeline;
  etna::Image            skyboxTexture;
  etna::Image            cupTexture;
};
