#pragma once
#include <glad/glad.h>
#include <glm/glm.hpp>
#include "../sim_api.h"
#include "mesh_builder.h"
#include "ibl_precompute.h"

struct ShaderProgram {
    GLuint program_id;
    GLint  uniform_projection;
    GLint  uniform_view;
    GLint  uniform_camera_pos;
    GLint  uniform_tex_array;
    // IBL (split-sum) inputs
    GLint  uniform_diffuse_irradiance;
    GLint  uniform_glossy_irradiance;
    GLint  uniform_brdf_lut;
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

    // Baked IBL cubemaps + BRDF lookup texture
    IBLMaps ibl;
    GLuint  brdf_lut_tex;

    // Texture array: rock layers + OBJ layers
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

// Resolve a path relative to the executable's directory (shaders, textures, …)
void resolve_exe_relative(const char* relative, char* out, size_t out_size);