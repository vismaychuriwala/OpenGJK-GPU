#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#include <libgen.h>
#endif
#include "opengl_renderer.h"
#include "../sim_config.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"

// ============================================================================
// Helpers
// ============================================================================

// Resolve a relative path against the executable's directory so that shader
// loading works regardless of the current working directory.
static void resolve_exe_relative(const char* relative, char* out, size_t out_size) {
#ifdef _WIN32
    char exe_path[MAX_PATH];
    GetModuleFileNameA(NULL, exe_path, MAX_PATH);
    // Strip filename to get directory
    char* last_sep = strrchr(exe_path, '\\');
    if (!last_sep) last_sep = strrchr(exe_path, '/');
    if (last_sep) *(last_sep + 1) = '\0';
    else exe_path[0] = '\0';
    snprintf(out, out_size, "%s%s", exe_path, relative);
#else
    char exe_path[1024];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len > 0) {
        exe_path[len] = '\0';
        char* dir = dirname(exe_path);
        snprintf(out, out_size, "%s/%s", dir, relative);
    } else {
        snprintf(out, out_size, "%s", relative);
    }
#endif
}

// ============================================================================
// Shaders
// ============================================================================

GLuint load_shader(const char* path, GLenum shader_type) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open shader: %s\n", path); return 0; }
    fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
    char* src = (char*)malloc(len + 1);
    size_t read_bytes = fread(src, 1, len, f);
    fclose(f);
    if ((long)read_bytes != len) {
        fprintf(stderr, "Shader partial read (%s): got %zu of %ld bytes\n", path, read_bytes, len);
        free(src);
        return 0;
    }
    src[len] = '\0';

    GLuint shader = glCreateShader(shader_type);
    glShaderSource(shader, 1, (const char**)&src, NULL);
    glCompileShader(shader);
    free(src);

    GLint ok; glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512]; glGetShaderInfoLog(shader, 512, NULL, log);
        fprintf(stderr, "Shader compile error (%s):\n%s\n", path, log);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

GLuint create_shader_program(const char* vert_path, const char* frag_path) {
    GLuint vs = load_shader(vert_path, GL_VERTEX_SHADER);
    GLuint fs = load_shader(frag_path, GL_FRAGMENT_SHADER);
    if (!vs || !fs) {
        if (vs) glDeleteShader(vs);
        if (fs) glDeleteShader(fs);
        return 0;
    }

    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs); glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs); glDeleteShader(fs);

    GLint ok; glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512]; glGetProgramInfoLog(prog, 512, NULL, log);
        fprintf(stderr, "Shader link error:\n%s\n", log);
        glDeleteProgram(prog);
        return 0;
    }
    return prog;
}

// ============================================================================
// Draw indirect command layout (matches GL spec)
// ============================================================================

struct DrawCmd {
    unsigned int count;
    unsigned int instanceCount;
    unsigned int firstIndex;
    int          baseVertex;
    unsigned int baseInstance;
};

// ============================================================================
// Static SSBO layout (must match shader)
// ============================================================================

struct GPUObjectStatic {
    float scale[3];
    float tex_index;   // -1 = flat color, 0 = polytope rock texture
    float color[4];
};  // 32 bytes

// ============================================================================
// renderer_init
// ============================================================================

bool renderer_init(OpenGLRenderer* renderer,
                   const MeshAtlas* atlas,
                   const ObjectInitData* objects,
                   int num_objects,
                   GLuint dynamic_pos_buffer,
                   GLuint dynamic_quat_buffer,
                   const char** obj_tex_paths,
                   int n_obj_tex)
{
    memset(renderer, 0, sizeof(OpenGLRenderer));
    renderer->num_objects         = num_objects;
    renderer->dynamic_pos_buffer  = dynamic_pos_buffer;
    renderer->dynamic_quat_buffer = dynamic_quat_buffer;

    // --- Shaders (resolve paths relative to executable) ---
    char path_buf[4][1024];
    resolve_exe_relative("shaders/object.vert", path_buf[0], sizeof(path_buf[0]));
    resolve_exe_relative("shaders/object.frag", path_buf[1], sizeof(path_buf[1]));
    resolve_exe_relative("shaders/ground.vert", path_buf[2], sizeof(path_buf[2]));
    resolve_exe_relative("shaders/ground.frag", path_buf[3], sizeof(path_buf[3]));

    renderer->object_shader.program_id =
        create_shader_program(path_buf[0], path_buf[1]);
    renderer->ground_shader.program_id =
        create_shader_program(path_buf[2], path_buf[3]);

    if (!renderer->object_shader.program_id || !renderer->ground_shader.program_id) {
        fprintf(stderr, "Failed to load shaders\n");
        return false;
    }

    renderer->object_shader.uniform_projection =
        glGetUniformLocation(renderer->object_shader.program_id, "uProjection");
    renderer->object_shader.uniform_view =
        glGetUniformLocation(renderer->object_shader.program_id, "uView");
    renderer->object_shader.uniform_light_dir =
        glGetUniformLocation(renderer->object_shader.program_id, "uLightDir");
    renderer->object_shader.uniform_camera_pos =
        glGetUniformLocation(renderer->object_shader.program_id, "uCameraPos");
    renderer->object_shader.uniform_env_map =
        glGetUniformLocation(renderer->object_shader.program_id, "uEnvMap");
    renderer->object_shader.uniform_has_env_map =
        glGetUniformLocation(renderer->object_shader.program_id, "uHasEnvMap");
    renderer->object_shader.uniform_tex_array =
        glGetUniformLocation(renderer->object_shader.program_id, "uTexArray");

    renderer->ground_shader.uniform_projection =
        glGetUniformLocation(renderer->ground_shader.program_id, "uProjection");
    renderer->ground_shader.uniform_view =
        glGetUniformLocation(renderer->ground_shader.program_id, "uView");

    // --- Geometry atlas upload ---
    glGenBuffers(1, &renderer->geometry_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, renderer->geometry_vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 atlas->num_vertices * sizeof(AtlasVertex),
                 atlas->vertices, GL_STATIC_DRAW);

    glGenBuffers(1, &renderer->geometry_ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, renderer->geometry_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 atlas->num_indices * sizeof(unsigned int),
                 atlas->indices, GL_STATIC_DRAW);

    // --- VAO: vertex format only, no per-instance attribs ---
    glGenVertexArrays(1, &renderer->mesh_vao);
    glBindVertexArray(renderer->mesh_vao);

    glBindBuffer(GL_ARRAY_BUFFER, renderer->geometry_vbo);
    // location 0: position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(AtlasVertex), (void*)0);
    glEnableVertexAttribArray(0);
    // location 1: normal
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(AtlasVertex), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // location 2: uv
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(AtlasVertex), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, renderer->geometry_ebo);
    glBindVertexArray(0);

    // --- Static SSBO ---
    GPUObjectStatic* static_data = (GPUObjectStatic*)malloc(num_objects * sizeof(GPUObjectStatic));
    for (int i = 0; i < num_objects; i++) {
        static_data[i].scale[0]   = objects[i].scale[0];
        static_data[i].scale[1]   = objects[i].scale[1];
        static_data[i].scale[2]   = objects[i].scale[2];
        static_data[i].tex_index  = objects[i].tex_index;
        static_data[i].color[0] = objects[i].color[0];
        static_data[i].color[1] = objects[i].color[1];
        static_data[i].color[2] = objects[i].color[2];
        static_data[i].color[3] = objects[i].color[3];
    }

    glGenBuffers(1, &renderer->static_ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, renderer->static_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 num_objects * sizeof(GPUObjectStatic),
                 static_data, GL_STATIC_DRAW);
    free(static_data);

    // --- Draw command buffer ---
    DrawCmd* cmds = (DrawCmd*)malloc(num_objects * sizeof(DrawCmd));
    for (int i = 0; i < num_objects; i++) {
        int mesh_id = objects[i].mesh_id;
        const MeshEntry* m = &atlas->meshes[mesh_id];
        cmds[i].count         = (unsigned int)m->index_count;
        cmds[i].instanceCount = 1;
        cmds[i].firstIndex    = (unsigned int)m->first_index;
        cmds[i].baseVertex    = m->base_vertex;
        cmds[i].baseInstance  = 0;  // not needed, we use gl_DrawID
    }

    glGenBuffers(1, &renderer->draw_cmd_buffer);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, renderer->draw_cmd_buffer);
    glBufferData(GL_DRAW_INDIRECT_BUFFER,
                 num_objects * sizeof(DrawCmd),
                 cmds, GL_STATIC_DRAW);
    free(cmds);

    // --- Ground plane ---
    float boundary = COMPUTE_BOUNDARY(num_objects);
    float gy = -boundary;
    float ground_verts[] = {
        -boundary, gy, -boundary,
         boundary, gy, -boundary,
         boundary, gy,  boundary,
        -boundary, gy, -boundary,
         boundary, gy,  boundary,
        -boundary, gy,  boundary,
    };

    glGenVertexArrays(1, &renderer->ground_vao);
    glGenBuffers(1, &renderer->ground_vbo);
    glBindVertexArray(renderer->ground_vao);
    glBindBuffer(GL_ARRAY_BUFFER, renderer->ground_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(ground_verts), ground_verts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    // --- Sky shader + empty VAO ---
    {
        char sky_vert[1024], sky_frag[1024];
        resolve_exe_relative("shaders/sky.vert", sky_vert, sizeof(sky_vert));
        resolve_exe_relative("shaders/sky.frag", sky_frag, sizeof(sky_frag));
        renderer->sky_program = create_shader_program(sky_vert, sky_frag);
        if (renderer->sky_program) {
            renderer->sky_uniform_inv_proj_view =
                glGetUniformLocation(renderer->sky_program, "uInvProjView");
            renderer->sky_uniform_env_map =
                glGetUniformLocation(renderer->sky_program, "uEnvMap");
        }
        glGenVertexArrays(1, &renderer->sky_vao);
    }

    // --- Environment map (equirectangular HDR) ---
    renderer->env_map_tex = 0;
    if (sizeof(ENV_MAP) > sizeof("")) {
        char env_path[1024];
        resolve_exe_relative("env_maps/" ENV_MAP, env_path, sizeof(env_path));
        int w, h, nc;
        stbi_set_flip_vertically_on_load(true);
        float* data = stbi_loadf(env_path, &w, &h, &nc, 0);
        if (data) {
            glGenTextures(1, &renderer->env_map_tex);
            glBindTexture(GL_TEXTURE_2D, renderer->env_map_tex);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, w, h, 0, GL_RGB, GL_FLOAT, data);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glGenerateMipmap(GL_TEXTURE_2D);
            stbi_image_free(data);
            fprintf(stderr, "Loaded env map: %s (%dx%d)\n", ENV_MAP, w, h);
        } else {
            fprintf(stderr, "Warning: could not load env map: %s\n", env_path);
        }
    }

    // --- Texture array: rock layers (from config) + OBJ layers (from caller) ---
    {
        static const char* rock_files[] = { TEXTURE_ARRAY_FILES };
        int n_rock = (int)(sizeof(rock_files) / sizeof(rock_files[0]));
        int total_layers = n_rock + n_obj_tex;

        glGenTextures(1, &renderer->tex_array);
        glBindTexture(GL_TEXTURE_2D_ARRAY, renderer->tex_array);
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_SRGB8_ALPHA8,
                     TEX_ARRAY_DIM, TEX_ARRAY_DIM, total_layers,
                     0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        auto load_layer = [&](const char* rel, int layer) {
            char full[1024];
            resolve_exe_relative(rel, full, sizeof(full));
            int w, h, nc;
            stbi_set_flip_vertically_on_load(true);
            unsigned char* data = stbi_load(full, &w, &h, &nc, 4);
            if (!data) {
                fprintf(stderr, "Warning: could not load texture layer %d: %s\n", layer, full);
                return;
            }
            unsigned char* px = data;
            unsigned char* resized = nullptr;
            if (w != TEX_ARRAY_DIM || h != TEX_ARRAY_DIM) {
                resized = (unsigned char*)malloc(TEX_ARRAY_DIM * TEX_ARRAY_DIM * 4);
                for (int dy = 0; dy < TEX_ARRAY_DIM; dy++) {
                    for (int dx = 0; dx < TEX_ARRAY_DIM; dx++) {
                        int sx = dx * w / TEX_ARRAY_DIM;
                        int sy = dy * h / TEX_ARRAY_DIM;
                        const unsigned char* s = data + (sy * w + sx) * 4;
                        unsigned char*       d = resized + (dy * TEX_ARRAY_DIM + dx) * 4;
                        d[0]=s[0]; d[1]=s[1]; d[2]=s[2]; d[3]=s[3];
                    }
                }
                px = resized;
            }
            glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0,
                            0, 0, layer,
                            TEX_ARRAY_DIM, TEX_ARRAY_DIM, 1,
                            GL_RGBA, GL_UNSIGNED_BYTE, px);
            if (resized) free(resized);
            stbi_image_free(data);
            fprintf(stderr, "Loaded texture layer %d: %s (%dx%d)\n", layer, rel, w, h);
        };

        for (int i = 0; i < n_rock; i++) {
            char rel[512];
            snprintf(rel, sizeof(rel), "textures/%s", rock_files[i]);
            load_layer(rel, i);
        }
        for (int i = 0; i < n_obj_tex; i++) {
            char rel[512];
            snprintf(rel, sizeof(rel), "textures/%s", obj_tex_paths[i]);
            load_layer(rel, n_rock + i);
        }

        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
    }

    return true;
}

// ============================================================================
// renderer_cleanup
// ============================================================================

void renderer_cleanup(OpenGLRenderer* renderer) {
    glDeleteProgram(renderer->object_shader.program_id);
    glDeleteProgram(renderer->ground_shader.program_id);
    glDeleteVertexArrays(1, &renderer->mesh_vao);
    glDeleteVertexArrays(1, &renderer->ground_vao);
    glDeleteBuffers(1, &renderer->geometry_vbo);
    glDeleteBuffers(1, &renderer->geometry_ebo);
    glDeleteBuffers(1, &renderer->static_ssbo);
    glDeleteBuffers(1, &renderer->draw_cmd_buffer);
    glDeleteBuffers(1, &renderer->ground_vbo);
    if (renderer->env_map_tex) glDeleteTextures(1, &renderer->env_map_tex);
    if (renderer->tex_array)   glDeleteTextures(1, &renderer->tex_array);
    if (renderer->sky_program) glDeleteProgram(renderer->sky_program);
    glDeleteVertexArrays(1, &renderer->sky_vao);
    // dynamic_pos_buffer is owned by caller
}

// ============================================================================
// renderer_draw
// ============================================================================

void renderer_draw(OpenGLRenderer* renderer,
                   const glm::mat4& projection,
                   const glm::mat4& view)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Sky (fullscreen triangle behind everything)
    if (renderer->sky_program && renderer->env_map_tex) {
        glDepthMask(GL_FALSE);
        glDisable(GL_DEPTH_TEST);
        glUseProgram(renderer->sky_program);
        glm::mat4 inv_proj_view = glm::inverse(projection * glm::mat4(glm::mat3(view)));
        glUniformMatrix4fv(renderer->sky_uniform_inv_proj_view, 1, GL_FALSE, &inv_proj_view[0][0]);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, renderer->env_map_tex);
        glUniform1i(renderer->sky_uniform_env_map, 0);
        glBindVertexArray(renderer->sky_vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glDepthMask(GL_TRUE);
        glEnable(GL_DEPTH_TEST);
    }

    // Ground
    glUseProgram(renderer->ground_shader.program_id);
    glUniformMatrix4fv(renderer->ground_shader.uniform_projection, 1, GL_FALSE, &projection[0][0]);
    glUniformMatrix4fv(renderer->ground_shader.uniform_view,       1, GL_FALSE, &view[0][0]);
    glBindVertexArray(renderer->ground_vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    // Objects
    glUseProgram(renderer->object_shader.program_id);
    glUniformMatrix4fv(renderer->object_shader.uniform_projection, 1, GL_FALSE, &projection[0][0]);
    glUniformMatrix4fv(renderer->object_shader.uniform_view,       1, GL_FALSE, &view[0][0]);

    glm::vec3 light_dir(0.5f, 0.7f, 0.3f);
    if (renderer->object_shader.uniform_light_dir >= 0)
        glUniform3fv(renderer->object_shader.uniform_light_dir, 1, &light_dir[0]);

    // Extract camera world position from view matrix: pos = -(R^T * t)
    glm::vec3 cam_pos = -glm::transpose(glm::mat3(view)) * glm::vec3(view[3]);
    if (renderer->object_shader.uniform_camera_pos >= 0)
        glUniform3fv(renderer->object_shader.uniform_camera_pos, 1, &cam_pos[0]);

    // Env map (unit 0)
    if (renderer->object_shader.uniform_has_env_map >= 0)
        glUniform1i(renderer->object_shader.uniform_has_env_map, renderer->env_map_tex ? 1 : 0);
    if (renderer->env_map_tex) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, renderer->env_map_tex);
        if (renderer->object_shader.uniform_env_map >= 0)
            glUniform1i(renderer->object_shader.uniform_env_map, 0);
    }

    // Texture array (unit 1)
    if (renderer->tex_array) {
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D_ARRAY, renderer->tex_array);
        if (renderer->object_shader.uniform_tex_array >= 0)
            glUniform1i(renderer->object_shader.uniform_tex_array, 1);
    }

    // Bind SSBOs
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, renderer->static_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, renderer->dynamic_pos_buffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, renderer->dynamic_quat_buffer);

    // Draw all objects in one call
    glBindVertexArray(renderer->mesh_vao);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, renderer->draw_cmd_buffer);
    glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT,
                                nullptr, renderer->num_objects, 0);

    glBindVertexArray(0);
}
