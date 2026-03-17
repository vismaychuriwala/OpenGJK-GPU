#pragma once
#include <glad/glad.h>
#include <glm.hpp>
#include "../sim_api.h"
#include "mesh_builder.h"

struct ShaderProgram {
    GLuint program_id;
    GLint  uniform_projection;
    GLint  uniform_view;
    GLint  uniform_light_dir;
};

struct OpenGLRenderer {
    ShaderProgram object_shader;
    ShaderProgram ground_shader;

    // Geometry atlas
    GLuint geometry_vbo;
    GLuint geometry_ebo;
    GLuint mesh_vao;

    // Per-object data
    GLuint static_ssbo;     // scale(3)+pad(1) + color(4) per object, uploaded once
    // dynamic position buffer (gl_pos_buffer) is owned by the caller and
    // registered with CUDA; we receive it in renderer_init and store a copy

    GLuint dynamic_pos_buffer;   // float4 per object (xyz=world pos), written by CUDA each frame
    GLuint dynamic_quat_buffer;  // float4 per object (xyzw=quaternion), written by CUDA each frame

    // Draw indirect
    GLuint draw_cmd_buffer;
    int    num_objects;

    // Ground plane
    GLuint ground_vao;
    GLuint ground_vbo;
};

// dynamic_pos_buffer / dynamic_quat_buffer: the same GL buffers passed to sim_init (registered with CUDA)
bool renderer_init(OpenGLRenderer* renderer,
                   const MeshAtlas* atlas,
                   const ObjectInitData* objects,
                   int num_objects,
                   GLuint dynamic_pos_buffer,
                   GLuint dynamic_quat_buffer);

void renderer_cleanup(OpenGLRenderer* renderer);
void renderer_draw(OpenGLRenderer* renderer,
                   const glm::mat4& projection,
                   const glm::mat4& view);

GLuint load_shader(const char* path, GLenum shader_type);
GLuint create_shader_program(const char* vert_path, const char* frag_path);
