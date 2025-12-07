#include "opengl_renderer.h"
#include "../sim_config.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// Icosahedron geometry (template)
static const float PHI = 1.618033988749895f;  // Golden ratio

// 12 vertices of icosahedron (normalized)
static const float icosahedron_vertices[] = {
    -1.0f,  PHI,  0.0f,
     1.0f,  PHI,  0.0f,
    -1.0f, -PHI,  0.0f,
     1.0f, -PHI,  0.0f,

     0.0f, -1.0f,  PHI,
     0.0f,  1.0f,  PHI,
     0.0f, -1.0f, -PHI,
     0.0f,  1.0f, -PHI,

     PHI,  0.0f, -1.0f,
     PHI,  0.0f,  1.0f,
    -PHI,  0.0f, -1.0f,
    -PHI,  0.0f,  1.0f,
};

// 20 triangular faces (indices into vertex array)
static const unsigned int icosahedron_triangles[] = {
    0, 11, 5,   0, 5, 1,    0, 1, 7,    0, 7, 10,   0, 10, 11,
    1, 5, 9,    5, 11, 4,   11, 10, 2,  10, 7, 6,   7, 1, 8,
    3, 9, 4,    3, 4, 2,    3, 2, 6,    3, 6, 8,    3, 8, 9,
    4, 9, 5,    2, 4, 11,   6, 2, 10,   8, 6, 7,    9, 8, 1
};

// 30 edges (indices into vertex array)
static const unsigned int icosahedron_edges[] = {
    0, 1,   0, 5,   0, 7,   0, 10,  0, 11,
    1, 5,   1, 7,   1, 8,   1, 9,
    2, 3,   2, 4,   2, 6,   2, 10,  2, 11,
    3, 4,   3, 6,   3, 8,   3, 9,
    4, 5,   4, 9,   4, 11,
    5, 9,   5, 11,
    6, 7,   6, 8,   6, 10,
    7, 8,   7, 10,
    8, 9,
    10, 11
};

// Normalize icosahedron vertices to unit sphere
static void normalize_vertices(float* vertices, int count) {
    for (int i = 0; i < count; i++) {
        float x = vertices[i * 3 + 0];
        float y = vertices[i * 3 + 1];
        float z = vertices[i * 3 + 2];
        float len = std::sqrt(x*x + y*y + z*z);
        vertices[i * 3 + 0] = x / len;
        vertices[i * 3 + 1] = y / len;
        vertices[i * 3 + 2] = z / len;
    }
}

// Load shader from file
GLuint load_shader(const char* path, GLenum shader_type) {
    FILE* file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open shader file: %s\n", path);
        return 0;
    }

    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* source = (char*)malloc(length + 1);
    fread(source, 1, length, file);
    source[length] = '\0';
    fclose(file);

    GLuint shader = glCreateShader(shader_type);
    glShaderSource(shader, 1, (const char**)&source, NULL);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetShaderInfoLog(shader, 512, NULL, info_log);
        fprintf(stderr, "Shader compilation failed (%s):\n%s\n", path, info_log);
        free(source);
        return 0;
    }

    free(source);
    return shader;
}

// Create shader program from vertex and fragment shaders
GLuint create_shader_program(const char* vert_path, const char* frag_path) {
    GLuint vert_shader = load_shader(vert_path, GL_VERTEX_SHADER);
    GLuint frag_shader = load_shader(frag_path, GL_FRAGMENT_SHADER);

    if (!vert_shader || !frag_shader) {
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vert_shader);
    glAttachShader(program, frag_shader);
    glLinkProgram(program);

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetProgramInfoLog(program, 512, NULL, info_log);
        fprintf(stderr, "Shader program linking failed:\n%s\n", info_log);
        return 0;
    }

    glDeleteShader(vert_shader);
    glDeleteShader(frag_shader);

    return program;
}

bool renderer_init(OpenGLRenderer* renderer, int max_objects) {
    std::memset(renderer, 0, sizeof(OpenGLRenderer));
    renderer->num_instances = max_objects;
    renderer->wireframe_enabled = true;

    // Load shaders
    renderer->sphere_solid_shader.program_id =
        create_shader_program("shaders/sphere_solid.vert", "shaders/sphere_solid.frag");
    renderer->sphere_wireframe_shader.program_id =
        create_shader_program("shaders/sphere_wireframe.vert", "shaders/sphere_wireframe.frag");
    renderer->ground_shader.program_id =
        create_shader_program("shaders/ground.vert", "shaders/ground.frag");

    if (!renderer->sphere_solid_shader.program_id ||
        !renderer->sphere_wireframe_shader.program_id ||
        !renderer->ground_shader.program_id) {
        fprintf(stderr, "Failed to load shaders\n");
        return false;
    }

    // Get uniform locations for solid shader
    renderer->sphere_solid_shader.uniform_projection =
        glGetUniformLocation(renderer->sphere_solid_shader.program_id, "uProjection");
    renderer->sphere_solid_shader.uniform_view =
        glGetUniformLocation(renderer->sphere_solid_shader.program_id, "uView");
    renderer->sphere_solid_shader.uniform_light_dir =
        glGetUniformLocation(renderer->sphere_solid_shader.program_id, "uLightDir");

    // Get uniform locations for wireframe shader
    renderer->sphere_wireframe_shader.uniform_projection =
        glGetUniformLocation(renderer->sphere_wireframe_shader.program_id, "uProjection");
    renderer->sphere_wireframe_shader.uniform_view =
        glGetUniformLocation(renderer->sphere_wireframe_shader.program_id, "uView");

    // Get uniform locations for ground shader
    renderer->ground_shader.uniform_projection =
        glGetUniformLocation(renderer->ground_shader.program_id, "uProjection");
    renderer->ground_shader.uniform_view =
        glGetUniformLocation(renderer->ground_shader.program_id, "uView");

    // Normalize icosahedron vertices
    float normalized_vertices[36];
    std::memcpy(normalized_vertices, icosahedron_vertices, sizeof(icosahedron_vertices));
    normalize_vertices(normalized_vertices, 12);

    // Compute normals (for icosahedron, normals = normalized vertex positions)
    float normals[36];
    std::memcpy(normals, normalized_vertices, sizeof(normalized_vertices));

    // Create vertex buffer for icosahedron positions
    glGenBuffers(1, &renderer->sphere_vertex_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, renderer->sphere_vertex_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(normalized_vertices), normalized_vertices, GL_STATIC_DRAW);

    // Create vertex buffer for normals
    glGenBuffers(1, &renderer->sphere_normal_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, renderer->sphere_normal_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(normals), normals, GL_STATIC_DRAW);

    // Create element buffer for triangles
    renderer->sphere_triangle_count = 20 * 3;
    glGenBuffers(1, &renderer->sphere_solid_ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, renderer->sphere_solid_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(icosahedron_triangles),
                 icosahedron_triangles, GL_STATIC_DRAW);

    // Create element buffer for edges
    renderer->sphere_edge_count = 30 * 2;
    glGenBuffers(1, &renderer->sphere_wireframe_ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, renderer->sphere_wireframe_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(icosahedron_edges),
                 icosahedron_edges, GL_STATIC_DRAW);

    // Create instance buffer (position, radius, color per instance)
    // Layout: vec3 position, float radius, vec4 color = 8 floats per instance
    glGenBuffers(1, &renderer->instance_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, renderer->instance_vbo);
    glBufferData(GL_ARRAY_BUFFER, max_objects * 8 * sizeof(float), NULL, GL_DYNAMIC_DRAW);

    // Create VAO for solid spheres
    glGenVertexArrays(1, &renderer->sphere_vao_solid);
    glBindVertexArray(renderer->sphere_vao_solid);

    // Attribute 0: vertex position (template)
    glBindBuffer(GL_ARRAY_BUFFER, renderer->sphere_vertex_vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Attribute 1: vertex normal (template)
    glBindBuffer(GL_ARRAY_BUFFER, renderer->sphere_normal_vbo);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);

    // Attribute 2: instance position
    glBindBuffer(GL_ARRAY_BUFFER, renderer->instance_vbo);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(2);
    glVertexAttribDivisor(2, 1);  // Advance once per instance

    // Attribute 3: instance radius
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(3);
    glVertexAttribDivisor(3, 1);

    // Attribute 4: instance color
    glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(4 * sizeof(float)));
    glEnableVertexAttribArray(4);
    glVertexAttribDivisor(4, 1);

    // Bind element buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, renderer->sphere_solid_ebo);

    // Create VAO for wireframe spheres
    glGenVertexArrays(1, &renderer->sphere_vao_wireframe);
    glBindVertexArray(renderer->sphere_vao_wireframe);

    // Attribute 0: vertex position (template)
    glBindBuffer(GL_ARRAY_BUFFER, renderer->sphere_vertex_vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Attribute 2: instance position (skip attribute 1, not used in wireframe)
    glBindBuffer(GL_ARRAY_BUFFER, renderer->instance_vbo);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(2);
    glVertexAttribDivisor(2, 1);

    // Attribute 3: instance radius
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(3);
    glVertexAttribDivisor(3, 1);

    // Bind element buffer for edges
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, renderer->sphere_wireframe_ebo);

    // Create ground plane (large quad)
    // Use COMPUTE_BOUNDARY macro to match the simulation boundary
    // Boundary box goes from -boundary to +boundary in each axis
    float boundary = COMPUTE_BOUNDARY(NUM_OBJECTS);
    float ground_y = -boundary;  // Ground at bottom of boundary box
    float ground_vertices[] = {
        -boundary, ground_y, -boundary,
         boundary, ground_y, -boundary,
         boundary, ground_y,  boundary,
        -boundary, ground_y, -boundary,
         boundary, ground_y,  boundary,
        -boundary, ground_y,  boundary,
    };

    glGenVertexArrays(1, &renderer->ground_vao);
    glGenBuffers(1, &renderer->ground_vbo);

    glBindVertexArray(renderer->ground_vao);
    glBindBuffer(GL_ARRAY_BUFFER, renderer->ground_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(ground_vertices), ground_vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);

    return true;
}

void renderer_cleanup(OpenGLRenderer* renderer) {
    glDeleteProgram(renderer->sphere_solid_shader.program_id);
    glDeleteProgram(renderer->sphere_wireframe_shader.program_id);
    glDeleteProgram(renderer->ground_shader.program_id);

    glDeleteVertexArrays(1, &renderer->sphere_vao_solid);
    glDeleteVertexArrays(1, &renderer->sphere_vao_wireframe);
    glDeleteVertexArrays(1, &renderer->ground_vao);

    glDeleteBuffers(1, &renderer->sphere_vertex_vbo);
    glDeleteBuffers(1, &renderer->sphere_normal_vbo);
    glDeleteBuffers(1, &renderer->sphere_solid_ebo);
    glDeleteBuffers(1, &renderer->sphere_wireframe_ebo);
    glDeleteBuffers(1, &renderer->instance_vbo);
    glDeleteBuffers(1, &renderer->ground_vbo);
}

void renderer_update_instances(OpenGLRenderer* renderer,
                                const float* positions,
                                const float* radii,
                                const float* colors,
                                int count) {
    // Pack data: position (vec3), radius (float), color (vec4) = 8 floats per instance
    float* instance_data = (float*)malloc(count * 8 * sizeof(float));

    for (int i = 0; i < count; i++) {
        int dst_offset = i * 8;
        int pos_offset = i * 3;
        int color_offset = i * 4;

        instance_data[dst_offset + 0] = positions[pos_offset + 0];
        instance_data[dst_offset + 1] = positions[pos_offset + 1];
        instance_data[dst_offset + 2] = positions[pos_offset + 2];
        instance_data[dst_offset + 3] = radii[i];
        instance_data[dst_offset + 4] = colors[color_offset + 0];
        instance_data[dst_offset + 5] = colors[color_offset + 1];
        instance_data[dst_offset + 6] = colors[color_offset + 2];
        instance_data[dst_offset + 7] = colors[color_offset + 3];
    }

    glBindBuffer(GL_ARRAY_BUFFER, renderer->instance_vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, count * 8 * sizeof(float), instance_data);

    free(instance_data);
    renderer->num_instances = count;
}

void renderer_draw(OpenGLRenderer* renderer, const glm::mat4& projection, const glm::mat4& view) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Draw ground plane
    glUseProgram(renderer->ground_shader.program_id);
    glUniformMatrix4fv(renderer->ground_shader.uniform_projection, 1, GL_FALSE, &projection[0][0]);
    glUniformMatrix4fv(renderer->ground_shader.uniform_view, 1, GL_FALSE, &view[0][0]);

    glBindVertexArray(renderer->ground_vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    // Draw solid spheres
    glUseProgram(renderer->sphere_solid_shader.program_id);
    glUniformMatrix4fv(renderer->sphere_solid_shader.uniform_projection, 1, GL_FALSE, &projection[0][0]);
    glUniformMatrix4fv(renderer->sphere_solid_shader.uniform_view, 1, GL_FALSE, &view[0][0]);

    glm::vec3 light_dir(0.5f, 0.7f, 0.3f);
    if (renderer->sphere_solid_shader.uniform_light_dir >= 0) {
        glUniform3fv(renderer->sphere_solid_shader.uniform_light_dir, 1, &light_dir[0]);
    }

    glBindVertexArray(renderer->sphere_vao_solid);
    glDrawElementsInstanced(GL_TRIANGLES, renderer->sphere_triangle_count,
                            GL_UNSIGNED_INT, 0, renderer->num_instances);

    // Draw wireframe
    if (renderer->wireframe_enabled) {
        glUseProgram(renderer->sphere_wireframe_shader.program_id);
        glUniformMatrix4fv(renderer->sphere_wireframe_shader.uniform_projection, 1, GL_FALSE, &projection[0][0]);
        glUniformMatrix4fv(renderer->sphere_wireframe_shader.uniform_view, 1, GL_FALSE, &view[0][0]);

        glBindVertexArray(renderer->sphere_vao_wireframe);
        glDrawElementsInstanced(GL_LINES, renderer->sphere_edge_count,
                                GL_UNSIGNED_INT, 0, renderer->num_instances);
    }

    glBindVertexArray(0);
}

void renderer_toggle_wireframe(OpenGLRenderer* renderer) {
    renderer->wireframe_enabled = !renderer->wireframe_enabled;
}
