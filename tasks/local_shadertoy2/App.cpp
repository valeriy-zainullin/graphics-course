#include "App.hpp"

#include <iostream>

#include <etna/BlockingTransferHelper.hpp>
#include <etna/Etna.hpp>
#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/Sampler.hpp>
#include <etna/RenderTargetStates.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

App::App()
  : resolution{1280, 720}
  , useVsync{true}
{
  // First, we need to initialize Vulkan, which is not trivial because
  // extensions are required for just about anything.
  {
    // GLFW tells us which extensions it needs to present frames to the OS window.
    // Actually rendering anything to a screen is optional in Vulkan, you can
    // alternatively save rendered frames into files, send them over network, etc.
    // Instance extensions do not depend on the actual GPU, only on the OS.
    auto glfwInstExts = windowing.getRequiredVulkanInstanceExtensions();

    std::vector<const char*> instanceExtensions{glfwInstExts.begin(), glfwInstExts.end()};

    // We also need the swapchain device extension to get access to the OS
    // window from inside of Vulkan on the GPU.
    // Device extensions require HW support from the GPU.
    // Generally, in Vulkan, we call the GPU a "device" and the CPU/OS combination a "host."
    std::vector<const char*> deviceExtensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    // Etna does all of the Vulkan initialization heavy lifting.
    // You can skip figuring out how it works for now.
    etna::initialize(etna::InitParams{
      .applicationName = "Local Shadertoy",
      .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
      .instanceExtensions = instanceExtensions,
      .deviceExtensions = deviceExtensions,
      // Replace with an index if etna detects your preferred GPU incorrectly
      .physicalDeviceIndexOverride = {},
      .numFramesInFlight = 1,
    });
  }

  // Now we can create an OS window
  osWindow = windowing.createWindow(OsWindow::CreateInfo{
    .resolution = resolution,
  });

  // But we also need to hook the OS window up to Vulkan manually!
  {
    // First, we ask GLFW to provide a "surface" for the window,
    // which is an opaque description of the area where we can actually render.
    auto surface = osWindow->createVkSurface(etna::get_context().getInstance());

    // Then we pass it to Etna to do the complicated work for us
    vkWindow = etna::get_context().createWindow(etna::Window::CreateInfo{
      .surface = std::move(surface),
    });

    // And finally ask Etna to create the actual swapchain so that we can
    // get (different) images each frame to render stuff into.
    // Here, we do not support window resizing, so we only need to call this once.
    auto [w, h] = vkWindow->recreateSwapchain(etna::Window::DesiredProperties{
      .resolution = {resolution.x, resolution.y},
      .vsync = useVsync,
    });

    // Technically, Vulkan might fail to initialize a swapchain with the requested
    // resolution and pick a different one. This, however, does not occur on platforms
    // we support. Still, it's better to follow the "intended" path.
    resolution = {w, h};
  }

  etna::GlobalContext& context = etna::get_context();

  // Next, we need a magical Etna helper to send commands to the GPU.
  // How it is actually performed is not trivial, but we can skip this for now.
  commandManager = context.createPerFrameCmdMgr();

  sampler = etna::Sampler(etna::Sampler::CreateInfo{.name = "sampler"});

  // Многое будет из лекции, просто чуть-чуть постараюсь объяснить по-своему.
  initSkyboxPipeline();
  initMainItemPipeline();

}

void App::initSkyboxPipeline() {
  etna::create_program(
    "local_shadertoy2_skybox",
    {LOCAL_SHADERTOY2_SHADERS_ROOT "quad.vert.spv", LOCAL_SHADERTOY2_SHADERS_ROOT "proc.frag.spv"});

  skyboxPipeline = etna::get_context().getPipelineManager().createGraphicsPipeline(
    "local_shadertoy2_skybox",
    etna::GraphicsPipeline::CreateInfo{
      .fragmentShaderOutput =
        {
          .colorAttachmentFormats = {vk::Format::eB8G8R8A8Srgb},
        },
    });

  // Создаем класс изображения.
  //   eColorAttachment нужен для того, чтобы фрагментный шейдер мог
  //   записывать в картинку.
  etna::Image::CreateInfo skyboxTextureInfo {
    .extent = vk::Extent3D{
        /* uint32_t */ .width  = 512,
        /* uint32_t */ .height = 512,
        /* uint32_t */ .depth  = 1,
    },
    /* std::string_view */ .name = "skyboxTexture",
    .format = vk::Format::eB8G8R8A8Srgb,
    .imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment,
  };

  etna::GlobalContext& context = etna::get_context();
  skyboxTexture = context.createImage(skyboxTextureInfo);
}

void App::initMainItemPipeline() {
  // Думаю, здесь загружается файл со spirv-байткодом для GPU. Этот кроссплатформенный и
  //   кроссвендорный байткод затем компилируется под видеокарту.
  // etna::create_program("local_shadertoy2", {LOCAL_SHADERTOY2_SHADERS_ROOT "toy.comp.spv"}); // Remain of compute approach.
  etna::create_program(
    "local_shadertoy2_main",
    {LOCAL_SHADERTOY2_SHADERS_ROOT "quad.vert.spv", LOCAL_SHADERTOY2_SHADERS_ROOT "toy.frag.spv"}
  );

  etna::GlobalContext& context = etna::get_context();

  // Команды GPU будут, чтобы записать массивы или загрузить нашу программу. Они попадают в очередь
  //   vk::CommandBuffer cmd_buf внутри etna, видимо. GPU c ними взаимодействует. Он может проставлять
  //   им флаги всякие (сказали в лекции).
  // Pipeline для нас синоним шейдера.
  // Eсть шейдер -- программа, но просьбу исполнять мы еще не создали. Создадим позже.
  //   Эти операции внутри command buffer как раз для этого. А так же у нас могут быть
  //   массивы внутри шейдера, в которые мы хотим передать данные, как в примере
  //   samples/simple_compute. Мы еще сможем записать в эти массивы данные.

  // pipeline = context.getPipelineManager().createComputePipeline("local_shadertoy2", {}); // Remain of compute shader approach.
  mainItemPipeline = context.getPipelineManager().createGraphicsPipeline(
    "local_shadertoy2_main",
    etna::GraphicsPipeline::CreateInfo{
      .fragmentShaderOutput =
        {
          .colorAttachmentFormats = {vk::Format::eB8G8R8A8Srgb},
        },
    }
    );
  // Я зашел внутрь в vs code, у нас есть там некоторый shader manager, он ищет шейдер.
  //   Потому имя должно совпадать с тем, что выше, наверно.

  // Читаем изображение из файла.
  // https://github.com/nothings/stb/blob/2e2bef463a5b53ddf8bb788e25da6b8506314c08/stb_image.h#L143
  int width = 0;
  int height = 0;
  int ncomp = 0;
  // Requesting 4, because we specify RGBA each 8bits in the image format.
  //   4 components per pixel. 
  auto cupTextureData = static_cast<void*>(
    stbi_load(GRAPHICS_COURSE_RESOURCES_ROOT "/textures/test_tex_1.png", &width, &height, &ncomp, 4)
  );
  // std::cerr << "cup texture number of components is " << ncomp << '\n';
  if (cupTextureData == nullptr) {
    std::cerr << "Failed to load cup texture. " << stbi_failure_reason() << std::endl;
    std::terminate();
  }

  // std::cerr << "1.\n";

  // Создаем класс изображения.
  etna::Image::CreateInfo cupTextureInfo {
    .extent = vk::Extent3D{
        /* uint32_t */ .width  = static_cast<uint32_t>(width),
        /* uint32_t */ .height = static_cast<uint32_t>(height),
        /* uint32_t */ .depth  = 1,
    },
    /* std::string_view */ .name = "cupTexture",
    .format = vk::Format::eR8G8B8A8Unorm,
    .imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eStorage,
  };
  cupTexture = context.createImage(cupTextureInfo);

  // Загружаем на видеокарту.
  auto tmpCmdBuf = commandManager->acquireNext();
  ETNA_CHECK_VK_RESULT(tmpCmdBuf.begin(vk::CommandBufferBeginInfo{}));
  // Не забудем освободим за собой память! Картинку уже должны были
  //   скопировать к этому моменту.
  std::unique_ptr<etna::OneShotCmdMgr> oneTimeMgr = etna::get_context().createOneShotCmdMgr();
  // На инициализацию нам тоже нужен командный буфер, чтобы загрузить картинку в
  //   видеокарту.
  size_t pictureSizeBytes = width * height * 4 /* color bytes per pixel */;
  etna::BlockingTransferHelper({ static_cast<vk::DeviceSize>(pictureSizeBytes) }).uploadImage(
    *oneTimeMgr,
    cupTexture,
    0,
    0,
    std::span<std::byte>(static_cast<std::byte*>(cupTextureData), pictureSizeBytes)
  );
  stbi_image_free(cupTextureData);
  ETNA_CHECK_VK_RESULT(tmpCmdBuf.end());

  // std::cerr << "2.\n";
  
  tmpCmdBuf = commandManager->acquireNext();
  ETNA_CHECK_VK_RESULT(tmpCmdBuf.begin(vk::CommandBufferBeginInfo{}));
  etna::set_state(
    tmpCmdBuf,
    cupTexture.get(),
    vk::PipelineStageFlagBits2::eFragmentShader,
    {vk::AccessFlagBits2::eShaderRead},
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor
  );
  etna::flush_barriers(tmpCmdBuf);
  // std::cerr << "3.\n";
  ETNA_CHECK_VK_RESULT(tmpCmdBuf.end());
}

App::~App()
{
  ETNA_CHECK_VK_RESULT(etna::get_context().getDevice().waitIdle());
}

void App::run()
{
  while (!osWindow->isBeingClosed())
  {
    windowing.poll();

    drawFrame();
  }

  // We need to wait for the GPU to execute the last frame before destroying
  // all resources and closing the application.
  ETNA_CHECK_VK_RESULT(etna::get_context().getDevice().waitIdle());
}

void App::prepareSkybox(vk::CommandBuffer& cmdBuffer) {
  etna::ShaderProgramInfo shader = etna::get_shader_program("local_shadertoy2_skybox");

  etna::RenderTargetState state{
    cmdBuffer,
    {{}, {500, 500}},
    {{skyboxTexture.get(), skyboxTexture.getView({})}},
    {}
  };

  // Закидываем на исполнение в очередь команд наш шейдер.
  cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, skyboxPipeline.getVkPipeline());

  // Закидываем еще параметры, которые обычно проставляет shadertoy.
  static const std::chrono::time_point initial_time = std::chrono::high_resolution_clock::now();
  std::chrono::time_point now = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(initial_time - now);
  struct {
    uint32_t resolution_x;
    uint32_t resolution_y;
    float time;
  } parameters = {
    static_cast<uint32_t>(resolution.x),
    static_cast<uint32_t>(resolution.y),
    static_cast<float>(duration.count() / 1000.0)
  };

  cmdBuffer.draw(3, 1, 0, 0);

  // etna::flush_barriers(cmdBuffer);
  etna::set_state(
    cmdBuffer,
    skyboxTexture.get(),
    vk::PipelineStageFlagBits2::eFragmentShader,
    {vk::AccessFlagBits2::eShaderRead},
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor
  );

}

void App::prepareMainItem(vk::Image& backbuffer, vk::ImageView& backbufferView, vk::CommandBuffer& cmdBuffer) {
  // Здесь мы узнаем, какие привязки (bindings) ожидает шейдер. Он ожидает всякие массивы, которые
  //   мы объявляем в коде шейдера в директивах layout. Там как раз указывают номер binding-а
  //   layout(std30, binding = 0) buffer a { float A[]; }, например.
  etna::ShaderProgramInfo shader = etna::get_shader_program("local_shadertoy2_main");

  etna::RenderTargetState state{
    cmdBuffer,
    {{}, {resolution.x, resolution.y}},
    {{backbuffer, backbufferView}},
    {}
  };

  // Закидываем на исполнение в очередь команд наш шейдер.
  cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, mainItemPipeline.getVkPipeline());

  // Привязываем всякие данные, они попадут в шейдер.
  auto set = etna::create_descriptor_set(
    shader.getDescriptorLayoutId(0),
    cmdBuffer,
    { etna::Binding{0, cupTexture.genBinding(sampler.get(), vk::ImageLayout::eReadOnlyOptimal)},
      etna::Binding{1, skyboxTexture.genBinding(sampler.get(), vk::ImageLayout::eReadOnlyOptimal)}
    });
  vk::DescriptorSet vkSet = set.getVkSet();
  cmdBuffer.bindDescriptorSets(
    vk::PipelineBindPoint::eGraphics,
    mainItemPipeline.getVkPipelineLayout(),
    0,
    1,
    &vkSet,
    0,
    nullptr);

  // Закидываем еще параметры, которые обычно проставляет shadertoy.
  static const std::chrono::time_point initial_time = std::chrono::high_resolution_clock::now();
  std::chrono::time_point now = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(initial_time - now);
  struct {
    uint32_t resolution_x;
    uint32_t resolution_y;
    float time;
  } parameters = {
    static_cast<uint32_t>(resolution.x),
    static_cast<uint32_t>(resolution.y),
    static_cast<float>(duration.count() / 1000.0)
  };
  // Указываем, что закидываем константы для фрагментного шейдера.
  //   И здесь важно указать правильный тип, иначе константы не дойдут до шейдера.
  //   Внутри local_shadertoy2_main есть еще и вершинный шейдер, у него другой shader stage flag,
  //   а compute шейдера уже вообще нет в программе local_shadertoy2.
  cmdBuffer.pushConstants(
    mainItemPipeline.getVkPipelineLayout(),
    vk::ShaderStageFlagBits::eFragment,
    0,
    sizeof(parameters),
    &parameters
  );

  cmdBuffer.draw(3, 1, 0, 0);
}

void App::drawFrame()
{
  // First, get a command buffer to write GPU commands into.
  auto currentCmdBuf = commandManager->acquireNext();

  // Next, tell Etna that we are going to start processing the next frame.
  etna::begin_frame();

  // And now get the image we should be rendering the picture into.
  auto nextSwapchainImage = vkWindow->acquireNext();

  // When window is minimized, we can't render anything in Windows
  // because it kills the swapchain, so we skip frames in this case.
  if (nextSwapchainImage)
  {
    auto [backbuffer, backbufferView, backbufferAvailableSem] = *nextSwapchainImage;

    ETNA_CHECK_VK_RESULT(currentCmdBuf.begin(vk::CommandBufferBeginInfo{}));

    auto skyboxVkImg = skyboxTexture.get();
    auto skyboxVkImgView = skyboxTexture.getView({});
    prepareSkybox(currentCmdBuf);
    prepareMainItem(backbuffer, backbufferView, currentCmdBuf);

    // At the end of "rendering", we are required to change how the pixels of the
    // swpchain image are laid out in memory to something that is appropriate
    // for presenting to the window (while preserving the content of the pixels!).
    etna::set_state(
      currentCmdBuf,
      backbuffer,
      // This looks weird, but is correct. Ask about it later.
      vk::PipelineStageFlagBits2::eColorAttachmentOutput,
      {},
      vk::ImageLayout::ePresentSrcKHR,
      vk::ImageAspectFlagBits::eColor);
    // And of course flush the layout transition.
    etna::flush_barriers(currentCmdBuf);

    ETNA_CHECK_VK_RESULT(currentCmdBuf.end());

    // We are done recording GPU commands now and we can send them to be executed by the GPU.
    // Note that the GPU won't start executing our commands before the semaphore is
    // signalled, which will happen when the OS says that the next swapchain image is ready.
    auto renderingDone =
      commandManager->submit(std::move(currentCmdBuf), std::move(backbufferAvailableSem));

    // Finally, present the backbuffer the screen, but only after the GPU tells the OS
    // that it is done executing the command buffer via the renderingDone semaphore.
    const bool presented = vkWindow->present(std::move(renderingDone), backbufferView);

    if (!presented)
      nextSwapchainImage = std::nullopt;
  }

  etna::end_frame();

  // After a window us un-minimized, we need to restore the swapchain to continue rendering.
  if (!nextSwapchainImage && osWindow->getResolution() != glm::uvec2{0, 0})
  {
    auto [w, h] = vkWindow->recreateSwapchain(etna::Window::DesiredProperties{
      .resolution = {resolution.x, resolution.y},
      .vsync = useVsync,
    });
    ETNA_VERIFY((resolution == glm::uvec2{w, h}));
  }
}
