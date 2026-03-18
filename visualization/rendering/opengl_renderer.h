#pragma once
#include <glad/glad.h>
#include <glm/glm.hpp>
#include "../sim_api.h"
#include "mesh_builder.h"

struct ShaderProgram {
    GLuint program_id;
    GLint  uniform_projection;
    GLint  uniform_view;
    GLint  uniform_light_dir;
    GLint  uniform_camera_pos;
    GLint  uniform_env_map;
    GLint  uniform_has_env_map;
    GLint  uniform_tex_array;
};

struct OpenGLRenderer {
    ShaderProgram object_shader;
    ShaderProgram ground_shader;

    // Geometry atlas
    GLuint geometry_vbo;
    GLuint geometry_ebo;
    GLuint mesh_vao;

    // Per-object data
    GLuint static_ssbo;
    GLuint dynamic_pos_buffer;
    GLuint dynamic_quat_buffer;

    // Draw indirect
    GLuint draw_cmd_buffer;
    int    num_objects;

    // Ground plane
    GLuint ground_vao;
    GLuint ground_vbo;

    // Environment map (unit 0)
    GLuint env_map_tex;

    // Texture array: rock layers + OBJ layers (unit 1)
    GLuint tex_array;

    // Sky
    GLuint sky_program;
    GLint  sky_uniform_inv_proj_view;
    GLint  sky_uniform_env_map;
    GLuint sky_vao;
};

bool renderer_init(OpenGLRenderer* renderer,
                   const MeshAtlas* atlas,
                   const ObjectInitData* objects,
                   int num_objects,
                   GLuint dynamic_pos_buffer,
                   GLuint dynamic_quat_buffer,
                   const char** obj_tex_paths,
                   int n_obj_tex);

void renderer_cleanup(OpenGLRenderer* renderer);
void renderer_draw(OpenGLRenderer* renderer,
                   const glm::mat4& projection,
                   const glm::mat4& view);

GLuint load_shader(const char* path, GLenum shader_type);
GLuint create_shader_program(const char* vert_path, const char* frag_path);
