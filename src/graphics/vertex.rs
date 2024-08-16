#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    /// Position is in normalized device coordinates
    position: [f32; 3],
    tex_coords: [f32; 2],
}

impl Vertex {
    pub fn new(position: [f32; 3], tex_coords: [f32; 2]) -> Self {
        Vertex {
            position,
            tex_coords,
        }
    }

    pub fn position(&self) -> &[f32; 3] {
        &self.position
    }

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // Remember to change this if the Vertex struct changes
                // This attribute corresponds to the `position` field of `Vertex`
                wgpu::VertexAttribute {
                    offset: 0,          // Offset in bytes is zero as this is the first attribute
                    shader_location: 0, // This is 0 as in the shader we decorate `position` in `VertexInput` in the shader file with `@location(0)`
                    format: wgpu::VertexFormat::Float32x3,
                },
                // This attribute corresponds to the `tex_coords` field of `Vertex`
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress, // Offset is the size of the first attribute - remember to change this if the size of `position` changes
                    shader_location: 1, // This is 1 as in the shader we decorate `tex_coords` in `VertexInput` in the shader file with `@location(1)`
                    format: wgpu::VertexFormat::Float32x2,
                },
            ], // Can use the wgpu::vertex_attr_array! macro to make this easier later
        }
    }
}
