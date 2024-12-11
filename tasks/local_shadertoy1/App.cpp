#include "App.hpp"

#include <etna/Etna.hpp>
#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/Sampler.hpp>


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


  // TODO: Initialize any additional resources you require here!

  // Многое будет из лекции, просто чуть-чуть постараюсь объяснить по-своему.

  // Думаю, здесь загружается файл со spirv-байткодом для GPU. Этот кроссплатформенный и
  //   кроссвендорный байткод затем компилируется под видеокарту.
  etna::create_program("local_shadertoy", {LOCAL_SHADERTOY_SHADERS_ROOT "toy.comp.spv"});

  // Команды GPU будут, чтобы записать массивы или загрузить нашу программу. Они попадают в очередь
  //   vk::CommandBuffer cmd_buf внутри etnf, видимо. GPU c ними взаимодействует. Он может проставлять
  //   им флаги всякие (сказали в лекции).
  // Pipeline для нас синоним шейдера.
  // Eсть шейдер -- программа, но просьбу исполнять мы еще не создали. Создадим позже.
  //   Эти операции внутри command buffer как раз для этого. А так же у нас могут быть
  //   массивы внутри шейдера, в которые мы хотим передать данные, как в примере
  //   samples/simple_compute. Мы еще сможем записать в эти массивы данные.

  pipeline = context.getPipelineManager().createComputePipeline("local_shadertoy", {});
  // Я зашел внутрь в vs code, у нас есть там некоторый shader manager, он ищет шейдер.
  //   Потому имя должно совпадать с тем, что выше, наверно.


  // Мы не можем просто в картинку vk::Image backbuffer, которая дала нам ОС, писать
  //   на видеокарте. Нам нужно создать еще один картинку, ее заполнять. А потом
  //   скопировать в исходную картинку backbuffer.
  //   Картинку backbuffer дает нам операционная система. И напрямую туда нельзя писать,
  //   сделано ради безопасности. А на некоторых ОС можно. Потому решили не ломать
  //   домашку у половины студентов :)
  // Возможно, дело в том, что в зависимости от ОС может быть разный порядок пикселей.
  //   Т.к. у вулкана много разных порядков пикселей можно выбрать, показывали на лекции.
  //   И чтобы кашу пользователь не видел с перемешанными пикселями на некоторых запретили..

  tmp_image = context.createImage(etna::Image::CreateInfo {
    .extent     = vk::Extent3D { resolution.x, resolution.y, 1},
    .name       = "image", // Интересно, что это за имя, кто по этому имени называет.
    .format     = vk::Format::eR8G8B8A8Snorm,
    .imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc
  });

  sampler = etna::Sampler(etna::Sampler::CreateInfo{.name = "sampler"});
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
    {
      // First of all, we need to "initialize" th "backbuffer", aka the current swapchain
      // image, into a state that is appropriate for us working with it. The initial state
      // is considered to be "undefined" (aka "I contain trash memory"), by the way.
      // "Transfer" in vulkanese means "copy or blit".
      // Note that Etna sometimes calls this for you to make life simpler, read Etna's code!
      etna::set_state(
        currentCmdBuf,
        backbuffer,
        // We are going to use the texture at the transfer stage...
        vk::PipelineStageFlagBits2::eTransfer,
        // ...to transfer-write stuff into it...
        vk::AccessFlagBits2::eTransferWrite,
        // ...and want it to have the appropriate layout.
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageAspectFlagBits::eColor);
      // The set_state doesn't actually record any commands, they are deferred to
      // the moment you call flush_barriers.
      // As with set_state, Etna sometimes flushes on it's own.
      // Usually, flushes should be placed before "action", i.e. compute dispatches
      // and blit/copy operations.
      etna::flush_barriers(currentCmdBuf);


      // Здесь мы узнаем, какие привязки (bindings) ожидает шейдер. Он ожидает всякие массивы, которые
      //   мы объявляем в коде шейдера в директивах layout. Там как раз указывают номер binding-а
      //   layout(std30, binding = 0) buffer a { float A[]; }, например.
      etna::ShaderProgramInfo shaderComputeInfo = etna::get_shader_program("local_shadertoy");

      // Привязываем всякие данные, они попадут в шейдер.
      etna::DescriptorSet set = etna::create_descriptor_set(
        shaderComputeInfo.getDescriptorLayoutId(0),
        currentCmdBuf,
        {
          etna::Binding{0, tmp_image.genBinding(sampler.get(), vk::ImageLayout::eGeneral)},
        }
      );
      vk::DescriptorSet vkSet = set.getVkSet(); // Сделали из декрипторов etna дескрипторы вулкана.

      // https://vkguide.dev/docs/chapter-4/descriptors/

      // Закидываем на исполнение в очередь команд наш шейдер.
      currentCmdBuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.getVkPipeline());
      // Указываем, какие данные шейдеру нужны (наши привязки, которые подготовили).
      // VULKAN_HPP_INLINE void CommandBuffer::bindDescriptorSets( VULKAN_HPP_NAMESPACE::PipelineBindPoint     pipelineBindPoint,
      //                                                           VULKAN_HPP_NAMESPACE::PipelineLayout        layout,
      //                                                           uint32_t                                    firstSet,
      //                                                           uint32_t                                    descriptorSetCount,
      //                                                           const VULKAN_HPP_NAMESPACE::DescriptorSet * pDescriptorSets,
      //                                                           uint32_t                                    dynamicOffsetCount,
      //                                                           const uint32_t *                            pDynamicOffsets,
      //                                                           Dispatch const &                            d ) const VULKAN_HPP_NOEXCEPT
      // Говорим, что хотим посчитать, шейдер на посчитать сделали. Даем наш шейдер. Говорим, сколько привязок, кидаем сами привязки.
      //   Про никакие оффсеты ничего не знаем. Типо так..
      currentCmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline.getVkPipelineLayout(), 0, 1, &vkSet, 0, nullptr);

      // Проставляем картинке права доступа, что в нее можно и нужно писать.
      //   И заодно ставим порядок пикселей (layout) самый стандартный.
      etna::set_state(
        currentCmdBuf,
        tmp_image.get(),
        vk::PipelineStageFlagBits2::eComputeShader,
        vk::AccessFlagBits2::eShaderWrite,
        vk::ImageLayout::eGeneral,
        vk::ImageAspectFlagBits::eColor
      );

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
      currentCmdBuf.pushConstants(pipeline.getVkPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(parameters), &parameters);

      // Нужно поставить flush в очередь команд, чтобы последующие команды увидели, видимо.
      etna::flush_barriers(currentCmdBuf);

      // Даем команду выполнить шейдер (pipeline в терминах vulkan).
      //   Вот выше не зря сделали барьер, теперь эта команда увидит команды до нее..
      currentCmdBuf.dispatch(resolution.x / 32, resolution.y / 32, 1);

      // Будем надеяться, что посчиталось. Теперь надо перегнать кадр из временной
      //   картинки в тот буфер, который дала ОС.

      // Изменим права доступа к кадру, что теперь он подлежит передаче из видеокарты
      //   обратно в оперативную память. И изменим layout его пикселей на оптимальный
      //   для передачи.
      etna::set_state(
        currentCmdBuf,
        tmp_image.get(),
        vk::PipelineStageFlagBits2::eBlit,
        vk::AccessFlagBits2::eTransferRead,
        vk::ImageLayout::eTransferSrcOptimal,
        vk::ImageAspectFlagBits::eColor
      );

      // Опять делаем flush, чтобы не смотреть на спам в логах, что layout
      //   не совпадает с тем, который был замечен у картинки в последний раз.
      //   Мы не сделали барьер на очереди команд, потому изменения и не заметны..
      etna::flush_barriers(currentCmdBuf);

      vk::ImageBlit region = {
          .srcSubresource = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),
          .srcOffsets     = {{vk::Offset3D(0, 0, 0), vk::Offset3D(resolution.x, resolution.y, 1)}},
          .dstSubresource = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),
          .dstOffsets     = {{vk::Offset3D(0, 0, 0), vk::Offset3D(resolution.x, resolution.y, 1)}},
      };

      currentCmdBuf.blitImage(
        tmp_image.get(),
        vk::ImageLayout::eTransferSrcOptimal,
        backbuffer,
        vk::ImageLayout::eTransferDstOptimal,
        1,
        &region,
        vk::Filter::eLinear
      );


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
    }
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
