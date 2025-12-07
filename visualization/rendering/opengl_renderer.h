#ifndef OPENGL_RENDERER_H
#define OPENGL_RENDERER_H

#include <glad/glad.h>
#include <glm.hpp>

// Shader program handle
struct ShaderProgram {
    GLuint program_id;
    GLint uniform_projection;
    GLint uniform_view;
    GLint uniform_light_dir;
};

// Renderer state
struct OpenGLRenderer {
    // Shaders
    ShaderProgram sphere_solid_shader;
    ShaderProgram sphere_wireframe_shader;
    ShaderProgram ground_shader;

    // Sphere geometry (template icosahedron)
    GLuint sphere_vao_solid;
    GLuint sphere_vao_wireframe;
    GLuint sphere_vertex_vbo;
    GLuint sphere_normal_vbo;
    GLuint sphere_solid_ebo;
    GLuint sphere_wireframe_ebo;
    int sphere_triangle_count;
    int sphere_edge_count;

    // Instance data buffer (shared with CUDA later)
    GLuint instance_vbo;
    int num_instances;

    // Ground plane
    GLuint ground_vao;
    GLuint ground_vbo;

    // State
    bool wireframe_enabled;
};

// Initialize renderer (shaders, geometry, buffers)
bool renderer_init(OpenGLRenderer* renderer, int max_objects);

// Cleanup renderer resources
void renderer_cleanup(OpenGLRenderer* renderer);

// Update instance buffer with new data (CPU path, before interop)
void renderer_update_instances(OpenGLRenderer* renderer,
                                const float* positions,   // Array of vec3
                                const float* radii,       // Array of float
                                const float* colors,      // Array of vec4
                                int count);

// Render the scene
void renderer_draw(OpenGLRenderer* renderer, const glm::mat4& projection, const glm::mat4& view);

// Toggle wireframe rendering
void renderer_toggle_wireframe(OpenGLRenderer* renderer);

// Utility: Load shader from file
GLuint load_shader(const char* path, GLenum shader_type);

// Utility: Create shader program from vertex and fragment shaders
GLuint create_shader_program(const char* vert_path, const char* frag_path);

#endif // OPENGL_RENDERER_H
