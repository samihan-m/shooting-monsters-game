use fyrox_sound::{
    buffer::{SoundBufferResource, SoundBufferResourceExtension},
    context::SoundContext,
    engine::SoundEngine,
    source::SoundSourceBuilder,
};
use graphics::CENTER;
use rand::{rngs::ThreadRng, Rng};
use wgpu::util::DeviceExt;

use wgpu_text::{
    glyph_brush::{
        ab_glyph::FontRef, HorizontalAlign, Layout, OwnedSection, Section, Text, VerticalAlign,
    },
    BrushBuilder, TextBrush,
};
use winit::{
    event::*,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Icon, Window, WindowBuilder},
};

// TODO:
// Make a new repository with things moved into better places (i.e. not all in one file and with better structure so that adding new features is easier)
// After that,
// Make a Main Menu
// Make a Pause Menu
// Make the state-update-rate independent of the render rate (make separate threads? (this sounds like a nightmare but it might be worth investigating))

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

mod graphics;
mod level;
mod texture;

struct Target {
    /// Position is in normalized device coordinates
    position: cgmath::Vector2<f32>,
    /// Higher values means the monster will be larger in appearance
    scale: f32,
}

struct State<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    monster_texture_bind_group: wgpu::BindGroup,
    instances: [graphics::Instance; 2],
    instance_buffer: wgpu::Buffer,

    cursor_position: winit::dpi::PhysicalPosition<f64>,
    monster: Target,
    rng: ThreadRng,

    crosshair_texture_bind_group: wgpu::BindGroup,

    sound_context: fyrox_sound::context::SoundContext,
    gunshot_buffer: fyrox_sound::buffer::SoundBufferResource,
    screech_buffer: fyrox_sound::buffer::SoundBufferResource,

    start_time: Option<web_time::Instant>,
    hit_count: u32,

    brush: TextBrush<FontRef<'a>>,
    time_remaining_text_section: OwnedSection,
    hit_count_text_section: OwnedSection,

    game_over_text_section: OwnedSection,

    instructions_text_section: OwnedSection,

    level_text_section: OwnedSection,
    you_win_text_section: OwnedSection,
    levels: Vec<level::Level>,
    current_level: usize,
    // Add new fields here as needed

    // Declaring window after surface is important to ensure surface is dropped first (as it contains unsafe references to the window's resources)
    window: &'a Window,
}

fn get_square_vertices(
    surface_width: u32,
    surface_height: u32,
    scale: f32,
) -> [graphics::Vertex; 4] {
    let ratio = surface_width as f32 / surface_height as f32;
    // The way this is implemented means that the width of the window has 0 impact on the size of the square, only the height does
    // If that's a problem, change this

    [
        graphics::Vertex::new([-scale / ratio, -scale], [0.0, 1.0]), // Bottom left
        graphics::Vertex::new([scale / ratio, -scale], [1.0, 1.0]),  // Bottom right
        graphics::Vertex::new([scale / ratio, scale], [1.0, 0.0]),   // Top right
        graphics::Vertex::new([-scale / ratio, scale], [0.0, 0.0]),  // Top left
    ]
}

const INITIAL_ROUND_LENGTH_SECONDS: f64 = 60.0;
const ROUND_LENGTH_DECREMENT: f64 = 5.0;
const INITIAL_REQUIRED_HIT_COUNT: u32 = 15;
const REQUIRED_HIT_COUNT_INCREMENT: u32 = 0;
const NUMBER_OF_LEVELS: u32 = 10;

fn get_seconds_remaining_in_round(
    start_time: Option<web_time::Instant>,
    round_length_seconds: f64,
) -> f64 {
    match start_time {
        None => round_length_seconds,
        Some(start_time) => {
            let elapsed = web_time::Instant::now() - start_time;
            let elapsed_seconds = elapsed.as_secs_f64();
            round_length_seconds - elapsed_seconds
        }
    }
}

// The game: Start a X second timer. The user has to hit the target Y times or else they lose. If they succeed, they move to the next level. X decreases with each level.
// Keep a count of how many times they hit the target and display (Y - it) in the top right corner (timer is top left corner)
// The timer will start once they've hit the target once.
// Show Game Over after time runs out, or You Win if they complete all levels.

impl<'a> State<'a> {
    async fn new(window: &'a Window) -> State<'a> {
        let size = window.inner_size();

        // Acquire GPU handle
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            // Because WebGPU is still experimental for some browsers, using GL as a fallback seems prudent.
            // Can probably change this to WebGPU and see what happens if we want
            backends: wgpu::Backends::GL,
            ..Default::default()
        });

        let surface = instance.create_surface(window).unwrap();

        // Acquire graphics card handle
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap(); // Will panic is no adapter with the required permissions is found

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: if cfg!(target_arch = "wasm32") {
                        // WebGL doesn't support all of wgpu's features, so disable them
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .unwrap();

        let surface_capabilities = surface.get_capabilities(&adapter);
        let surface_format = surface_capabilities
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_capabilities.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_capabilities.present_modes[0], // For Vsync, use Fifo. It is always supported
            alpha_mode: surface_capabilities.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let monster_texture =
            texture::Texture::from_bytes(&device, &queue, include_bytes!("eyes.png"), "eyes.png")
                .unwrap();

        let crosshair_texture = texture::Texture::from_bytes(
            &device,
            &queue,
            include_bytes!("crosshair.png"),
            "crosshair.png",
        )
        .unwrap();

        // We are storing the bind group separately so that `Texture` doesn't need to know how the `BindGroup` is laid out.

        // A BindGroup describes a set of resources and how they can be accessed by a shader
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0, // This should match the @binding(0) for the `texture_2d` in the shader file
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1, // This should match the @binding(1) for the `sampler` in the shader file
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), // This should match the `filterable` field of the corresponding `Texture` entry above
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });
        // Defining a `BindGroupLayout` separately allows us to swap out `BindGroup`s that share the same layout

        let monster_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&monster_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&monster_texture.sampler),
                },
            ],
            label: Some("target_bind_group"),
        });

        let crosshair_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&crosshair_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&crosshair_texture.sampler),
                },
            ],
            label: Some("other_bind_group"),
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main", // Remember to change this if we change the name of the vertex shader function
                buffers: &[graphics::Vertex::desc(), graphics::Instance::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main", // Remember to change this if we change the name of the fragment shader function
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                // Ccw means a triangle is facing forward if the vertices are arranged in a CCW direction.
                // Triangles that are not facing forward are culled (not included in the render) as specified by cull_mode
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // When committing to 2D, this can be set to None
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false, // This is related to anti-aliasing
            },
            multiview: None, // Used when rendering to array textures
            cache: None, // Enables caching shader compilation data - only really useful for Android build targets
        });

        let monster_scale = 0.25;
        let vertices = get_square_vertices(config.width, config.height, monster_scale);

        let num_triangles = vertices.len() as u16 - 2;
        let indices = (1u16..=num_triangles)
            .flat_map(|i| vec![i + 1, i, 0])
            .collect::<Vec<_>>();
        let num_indices = indices.len() as u32;

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let instances = [
            graphics::Instance { position: CENTER }, // The target
            graphics::Instance { position: CENTER }, // The crosshair
        ];

        let instance_data = instances
            .iter()
            .map(graphics::Instance::to_raw)
            .collect::<Vec<_>>();

        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let cursor_position = winit::dpi::PhysicalPosition::new(0.0, 0.0);

        let monster = Target {
            position: CENTER,
            scale: monster_scale,
        };

        let rng = rand::thread_rng();

        let sound_engine = SoundEngine::new().unwrap();
        let sound_context = SoundContext::new();
        sound_engine.state().add_context(sound_context.clone());

        let gunshot_buffer = SoundBufferResource::new_generic(
            fyrox_sound::buffer::DataSource::from_memory(include_bytes!("shot.wav").to_vec()),
        )
        .unwrap();

        let screech_buffer = SoundBufferResource::new_generic(
            fyrox_sound::buffer::DataSource::from_memory(include_bytes!("screech.wav").to_vec()),
        )
        .unwrap();

        let start_time = None;
        let hit_count = 0;

        let font = include_bytes!("font.ttf");
        let brush = BrushBuilder::using_font_bytes(font).unwrap().build(
            &device,
            config.width,
            config.height,
            config.format,
        );
        let font_size = 25.;
        let time_remaining_text_section = Section::default()
            .add_text(
                Text::new(&format!("Time remaining: {}", INITIAL_ROUND_LENGTH_SECONDS))
                    .with_scale(font_size)
                    .with_color([1.0, 0.0, 0.0, 1.0]),
            )
            .with_bounds([config.width as f32, config.height as f32])
            .with_layout(
                Layout::default()
                    .v_align(VerticalAlign::Top)
                    .h_align(HorizontalAlign::Left)
                    .line_breaker(wgpu_text::glyph_brush::BuiltInLineBreaker::AnyCharLineBreaker),
            )
            .with_screen_position((0.5 * config.width as f32, 0.5 * config.height as f32))
            .to_owned();

        let hit_count_text_section = Section::default()
            .add_text(
                Text::new("Hits: 0")
                    .with_scale(font_size)
                    .with_color([1.0, 0.0, 0.0, 1.0]),
            )
            .with_bounds([config.width as f32, config.height as f32])
            .with_layout(
                Layout::default()
                    .v_align(VerticalAlign::Top)
                    .h_align(HorizontalAlign::Right)
                    .line_breaker(wgpu_text::glyph_brush::BuiltInLineBreaker::AnyCharLineBreaker),
            )
            .with_screen_position((config.width as f32, 0.))
            .to_owned();

        let game_over_text_section = Section::default()
            .add_text(
                Text::new("Game Over")
                    .with_scale(50.)
                    .with_color([1.0, 0.0, 0.0, 1.0]),
            )
            .add_text(
                Text::new("\nPress Space to restart")
                    .with_scale(25.)
                    .with_color([1.0, 0.0, 0.0, 1.0]),
            )
            .with_bounds([config.width as f32, config.height as f32])
            .with_layout(
                Layout::default()
                    .v_align(VerticalAlign::Center)
                    .h_align(HorizontalAlign::Center)
                    .line_breaker(wgpu_text::glyph_brush::BuiltInLineBreaker::AnyCharLineBreaker),
            )
            .with_screen_position((config.width as f32 / 2., config.height as f32 / 2.))
            .to_owned();

        let instructions_text_section = Section::default()
            .add_text(
                Text::new("Left-click to shoot.")
                    .with_scale(50.)
                    .with_color([1.0, 0.0, 0.0, 1.0]),
            )
            .add_text(
                Text::new("\nThe timer will start after you hit the target once.")
                    .with_scale(25.)
                    .with_color([1.0, 0.0, 0.0, 1.0]),
            )
            .with_bounds([config.width as f32, config.height as f32])
            .with_layout(
                Layout::default()
                    .v_align(VerticalAlign::Center)
                    .h_align(HorizontalAlign::Center)
                    .line_breaker(wgpu_text::glyph_brush::BuiltInLineBreaker::AnyCharLineBreaker),
            )
            .with_screen_position((config.width as f32 / 2., config.height as f32 / 1.5))
            .to_owned();

        let level_text_section = Section::default()
            .add_text(
                Text::new("Level: 1")
                    .with_scale(25.)
                    .with_color([1.0, 0.0, 0.0, 1.0]),
            )
            .with_bounds([config.width as f32, config.height as f32])
            .with_layout(
                Layout::default()
                    .v_align(VerticalAlign::Top)
                    .h_align(HorizontalAlign::Center)
                    .line_breaker(wgpu_text::glyph_brush::BuiltInLineBreaker::AnyCharLineBreaker),
            )
            .with_screen_position((config.width as f32 / 2., 0.))
            .to_owned();

        let you_win_text_section = Section::default()
            .add_text(
                Text::new("You Win!")
                    .with_scale(50.)
                    .with_color([1.0, 0.0, 0.0, 1.0]),
            )
            .add_text(
                Text::new("\nPress Space to restart")
                    .with_scale(25.)
                    .with_color([1.0, 0.0, 0.0, 1.0]),
            )
            .with_bounds([config.width as f32, config.height as f32])
            .with_layout(
                Layout::default()
                    .v_align(VerticalAlign::Center)
                    .h_align(HorizontalAlign::Center)
                    .line_breaker(wgpu_text::glyph_brush::BuiltInLineBreaker::AnyCharLineBreaker),
            )
            .with_screen_position((config.width as f32 / 2., config.height as f32 / 2.))
            .to_owned();

        let levels = level::Level::make_level_list(
            INITIAL_ROUND_LENGTH_SECONDS,
            ROUND_LENGTH_DECREMENT,
            INITIAL_REQUIRED_HIT_COUNT,
            REQUIRED_HIT_COUNT_INCREMENT,
            NUMBER_OF_LEVELS,
        );
        let current_level = 0;

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            monster_texture_bind_group,
            instances,
            instance_buffer,

            cursor_position,
            monster,
            rng,

            crosshair_texture_bind_group,

            sound_context,
            gunshot_buffer,
            screech_buffer,

            start_time,
            hit_count,

            brush,
            time_remaining_text_section,
            hit_count_text_section,

            game_over_text_section,

            instructions_text_section,

            level_text_section,
            you_win_text_section,
            levels,
            current_level,

            window,
        }
    }

    pub fn window(&self) -> &Window {
        self.window
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            // Update the text sections
            // Without this code, the text will start to get squished as the window is resized
            // Also without this code (I think? I have no other explanation) the text will not render correctly in a WASM environment (it all gets squished to the top left corner (mostly even out of frame)).
            self.time_remaining_text_section.bounds =
                (new_size.width as f32, new_size.height as f32);
            self.time_remaining_text_section.screen_position = (0., 0.);
            self.hit_count_text_section.bounds = (new_size.width as f32, new_size.height as f32);
            self.hit_count_text_section.screen_position = (new_size.width as f32, 0.);
            self.game_over_text_section.bounds = (new_size.width as f32, new_size.height as f32);
            self.game_over_text_section.screen_position =
                (new_size.width as f32 / 2., new_size.height as f32 / 2.);
            self.instructions_text_section.bounds = (new_size.width as f32, new_size.height as f32);
            self.instructions_text_section.screen_position =
                (new_size.width as f32 / 2., new_size.height as f32 / 1.5);
            self.level_text_section.bounds = (new_size.width as f32, new_size.height as f32);
            self.level_text_section.screen_position = (new_size.width as f32 / 2., 0.);
            self.you_win_text_section.bounds = (new_size.width as f32, new_size.height as f32);
            self.you_win_text_section.screen_position =
                (new_size.width as f32 / 2., new_size.height as f32 / 2.);
            self.brush.resize_view(
                self.config.width as f32,
                self.config.height as f32,
                &self.queue,
            );

            // Update the vertices to match the new size
            let vertices = get_square_vertices(new_size.width, new_size.height, self.monster.scale);
            self.queue
                .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        // WindowEvents will be captured here - if we return false, the event will propagate to the main loop to be handled there as well
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_position = *position;
                true
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                let time_remaining = get_seconds_remaining_in_round(
                    self.start_time,
                    self.get_current_level_round_length_seconds(),
                );

                if time_remaining < 0.0 || self.current_level >= self.levels.len() {
                    return true;
                }

                // Ready the gunshot sound
                let gunshot_source = SoundSourceBuilder::new()
                    .with_buffer(self.gunshot_buffer.clone())
                    .with_status(fyrox_sound::source::Status::Playing)
                    .with_play_once(true)
                    .with_gain(0.25)
                    .build()
                    .unwrap();
                self.sound_context.state().add_source(gunshot_source);

                let is_cursor_on_monster = {
                    let cursor_ndc = cgmath::Vector2::new(
                        (self.cursor_position.x as f32 / self.size.width as f32) * 2.0 - 1.0,
                        -((self.cursor_position.y as f32 / self.size.height as f32) * 2.0 - 1.0),
                    );

                    let [bottom_left_vertex, _, top_right_vertex, _] =
                        get_square_vertices(self.size.width, self.size.height, self.monster.scale);

                    let [min_x, min_y] = bottom_left_vertex.position();
                    let [max_x, max_y] = top_right_vertex.position();
                    // These bounds are for the target if it was at the origin - they need to be adjusted to correspond to the instance position

                    let target_min_x = self.monster.position.x + min_x; // min_x is already negative so don't subtract it
                    let target_max_x = self.monster.position.x + max_x;
                    let target_min_y = self.monster.position.y + min_y; // min_y is already negative so don't subtract it
                    let target_max_y = self.monster.position.y + max_y;

                    cursor_ndc.x >= target_min_x
                        && cursor_ndc.x <= target_max_x
                        && cursor_ndc.y >= target_min_y
                        && cursor_ndc.y <= target_max_y
                };

                if is_cursor_on_monster {
                    // Play the screech noise
                    let screech_source = SoundSourceBuilder::new()
                        .with_buffer(self.screech_buffer.clone())
                        .with_status(fyrox_sound::source::Status::Playing)
                        .with_play_once(true)
                        .with_gain(0.35)
                        .build()
                        .unwrap();
                    self.sound_context.state().add_source(screech_source);

                    // Move target
                    let distribution = rand::distributions::Uniform::new(-1.0, 1.0);

                    self.monster.position = cgmath::Vector2::new(
                        self.rng.sample(distribution),
                        self.rng.sample(distribution),
                    );

                    if self.hit_count == 0 {
                        self.start_time = Some(web_time::Instant::now());
                    }
                    self.hit_count += 1;

                    // Check if the level is complete
                    if let Some(level) = self.levels.get(self.current_level) {
                        if self.hit_count >= level.required_hit_count() {
                            self.current_level += 1;
                            self.hit_count = 0;
                            self.start_time = None;
                        }
                    }
                }

                true
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Space),
                        ..
                    },
                ..
            } => {
                // If the game is over, reset the game
                if get_seconds_remaining_in_round(
                    self.start_time,
                    self.get_current_level_round_length_seconds(),
                ) < 0.0
                {
                    self.start_time = None;
                    self.hit_count = 0;
                    self.monster.position = CENTER;
                } else if self.current_level >= self.levels.len() {
                    // If the player has won, reset the game
                    self.start_time = None;
                    self.hit_count = 0;
                    self.monster.position = CENTER;
                    self.current_level = 0;
                }

                true
            }
            _ => false,
        }
    }

    fn update(&mut self) {
        // Update the text
        let time_remaining = get_seconds_remaining_in_round(
            self.start_time,
            self.get_current_level_round_length_seconds(),
        );
        self.time_remaining_text_section.text[0].text =
            format!("Time remaining: {}", time_remaining.round().max(0.)); // Am fine with indexing into text[0] as we know we have one text element (defined in ::new())
        let current_level = self.levels.get(self.current_level);
        match current_level {
            Some(level) => {
                self.hit_count_text_section.text[0].text = format!(
                    "Hits required: {}",
                    level.required_hit_count() - self.hit_count
                ); // Am fine with indexing into text[0] as we know we have one text element (defined in ::new())
                self.level_text_section.text[0].text = format!("Level: {}", self.current_level + 1);
                // Am fine with indexing into text[0] as we know we have one text element (defined in ::new())
            }
            None => {
                self.hit_count_text_section.text[0].text = "Hits required: 0".to_string();
                self.level_text_section.text[0].text = format!("Level: {}", self.current_level);
                // Am fine with indexing into text[0] as we know we have one text element (defined in ::new())
            }
        }

        // Update the instance buffer with the latest position of the target
        let target = &mut self.instances[0];
        target.position = self.monster.position;

        // Update the crosshair instance buffer with the latest cursor position
        let crosshair = &mut self.instances[1];
        let crosshair_ndc_position = cgmath::Vector2::new(
            (self.cursor_position.x as f32 / self.size.width as f32) * 2.0 - 1.0,
            -((self.cursor_position.y as f32 / self.size.height as f32) * 2.0 - 1.0),
        );
        crosshair.position = crosshair_ndc_position;

        let instance_data = self
            .instances
            .iter()
            .map(graphics::Instance::to_raw)
            .collect::<Vec<_>>();
        self.queue.write_buffer(
            &self.instance_buffer,
            0,
            bytemuck::cast_slice(&instance_data),
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Sending commands to the GPU requires a command buffer (and an encoder to write to that buffer)
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // This block is required so that a mutable borrow on `encoder` ends so we can call `finish()` on it later
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                // @location(0) in the fragment shader targets this first `RenderPassColorAttachment`
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.01,
                            g: 0.01,
                            b: 0.01,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

            // Draw target
            render_pass.set_bind_group(0, &self.monster_texture_bind_group, &[]);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);

            // Draw crosshair
            render_pass.set_bind_group(0, &self.crosshair_texture_bind_group, &[]);
            render_pass.draw_indexed(0..self.num_indices, 0, 1..2);

            // Draw text
            let mut sections = vec![
                &self.time_remaining_text_section,
                &self.hit_count_text_section,
                &self.level_text_section,
            ];

            if self.current_level >= self.levels.len() {
                // Show 'You Win!' message
                sections.push(&self.you_win_text_section);
            } else {
                match get_seconds_remaining_in_round(
                    self.start_time,
                    self.get_current_level_round_length_seconds(),
                ) {
                    time if time < 0.0 => sections.push(&self.game_over_text_section),
                    time if time == self.get_current_level_round_length_seconds() => {
                        sections.push(&self.instructions_text_section)
                    }
                    _ => {}
                }
            }

            let Ok(_) = self.brush.queue(&self.device, &self.queue, sections) else {
                panic!("Failed to queue text");
            };
            self.brush.draw(&mut render_pass);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn get_current_level_round_length_seconds(&self) -> f64 {
        match self.levels.get(self.current_level) {
            Some(level) => level.round_length_seconds(),
            None => 0.0,
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger");
        } else {
            // When wgpu hits an error, it panics with a generic message and logs the real error via the log crate.
            // Enabling logging like this will prevent wgpu from failing silently.
            env_logger::init();
        }
    }

    let event_loop = EventLoop::new().unwrap();
    let image = image::load_from_memory(include_bytes!("eyes.png"))
        .unwrap()
        .into_rgba8();
    let icon = Icon::from_rgba(image.to_vec(), image.width(), image.height()).unwrap();
    let window = WindowBuilder::new()
        .with_inner_size(winit::dpi::PhysicalSize::new(1280, 720))
        .with_title("Shooting Monsters")
        .with_window_icon(Some(icon.clone()))
        .build(&event_loop)
        .unwrap();
    window.set_cursor_visible(false);

    #[cfg(target_arch = "wasm32")]
    {
        // Winit prevents sizing with CSS, so we have to set
        // the size manually when on web.
        use winit::dpi::PhysicalSize;

        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.get_element_by_id("app")?;
                let canvas = web_sys::Element::from(window.canvas()?);
                dst.append_child(&canvas).ok()?;
                Some(())
            })
            .expect("Couldn't append canvas to document body.");

        let _ = window.request_inner_size(PhysicalSize::new(1280, 720));
    }

    let mut state = State::new(&window).await;
    let mut surface_configured = false;

    #[cfg(not(target_arch = "wasm32"))]
    let mut then = std::time::Instant::now();
    #[cfg(not(target_arch = "wasm32"))]
    let mut now = std::time::Instant::now();
    #[cfg(not(target_arch = "wasm32"))]
    let mut fps = 0u32;

    let _ = event_loop.run(move |event, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window().id() => {
                if !state.input(event) {
                    match event {
                        WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput {
                            event:
                                KeyEvent {
                                    state: ElementState::Pressed,
                                    physical_key: PhysicalKey::Code(KeyCode::Escape),
                                    ..
                                },
                            ..
                        } => control_flow.exit(),
                        WindowEvent::Resized(physical_size) => {
                            log::info!("physical_size: {:?}", physical_size);
                            surface_configured = true;
                            state.resize(*physical_size);
                        }
                        WindowEvent::RedrawRequested if window_id == state.window().id() => {
                            // Tell winit that we want another frame after this one
                            state.window().request_redraw();

                            if !surface_configured {
                                return;
                            }

                            state.update();
                            match state.render() {
                                Ok(_) => {}
                                // Reconfigure the surface if lost
                                Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                                    state.resize(state.size)
                                }
                                // Out of memory, probably best to exit
                                Err(wgpu::SurfaceError::OutOfMemory) => {
                                    log::error!("OutOfMemory");
                                    control_flow.exit();
                                }
                                Err(wgpu::SurfaceError::Timeout) => {
                                    // This happens when a frame takes too long to present
                                    log::warn!("Surface timeout")
                                }
                            }

                            #[cfg(not(target_arch = "wasm32"))]
                            {
                                fps += 1;
                                if now.duration_since(then) >= std::time::Duration::from_secs(1) {
                                    state.window().set_title(&format!("FPS: {}", fps));
                                    then = now;
                                    fps = 0;
                                }
                                now = std::time::Instant::now();
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    });
}
