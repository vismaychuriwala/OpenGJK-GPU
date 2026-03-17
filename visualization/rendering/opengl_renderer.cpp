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
    float pad;
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
                   GLuint dynamic_quat_buffer)
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

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, renderer->geometry_ebo);
    glBindVertexArray(0);

    // --- Static SSBO ---
    GPUObjectStatic* static_data = (GPUObjectStatic*)malloc(num_objects * sizeof(GPUObjectStatic));
    for (int i = 0; i < num_objects; i++) {
        static_data[i].scale[0] = objects[i].scale[0];
        static_data[i].scale[1] = objects[i].scale[1];
        static_data[i].scale[2] = objects[i].scale[2];
        static_data[i].pad      = 0.0f;
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
