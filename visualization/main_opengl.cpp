#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm.hpp>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <cmath>

#include "rendering/input.h"
#include "rendering/camera.h"
#include "rendering/opengl_renderer.h"
#include "rendering/mesh_builder.h"
#include "sim_config.h"
#include "sim_api.h"

static float colors[][4] = {
    {0.9f, 0.2f, 0.2f, 1.0f},
    {0.2f, 0.2f, 0.9f, 1.0f},
    {0.2f, 0.9f, 0.2f, 1.0f},
    {0.9f, 0.6f, 0.2f, 1.0f},
    {0.7f, 0.2f, 0.9f, 1.0f},
    {0.9f, 0.5f, 0.8f, 1.0f},
    {0.5f, 0.9f, 0.2f, 1.0f},
    {0.4f, 0.7f, 0.9f, 1.0f},
    {0.6f, 0.2f, 0.3f, 1.0f},
};
static const int NUM_COLORS = sizeof(colors) / sizeof(colors[0]);

// Shape types, one mesh_id per type (filled at build time)
enum ShapeType { SHAPE_ICOSAHEDRON = 0, SHAPE_BOX, SHAPE_TETRAHEDRON, SHAPE_OCTAHEDRON, SHAPE_COUNT };
static int g_mesh_ids[SHAPE_COUNT];

static float* build_gjk_verts(ShapeType type, int* out_count) {
    switch (type) {
        case SHAPE_ICOSAHEDRON:  return gen_gjk_icosahedron(out_count);
        case SHAPE_BOX:          return gen_gjk_box(out_count);
        case SHAPE_TETRAHEDRON:  return gen_gjk_tetrahedron(out_count);
        case SHAPE_OCTAHEDRON:   return gen_gjk_octahedron(out_count);
        default:                 return gen_gjk_icosahedron(out_count);
    }
}

static void build_scene(MeshAtlas* atlas, ObjectInitData* objects, int num_objects) {
    // Register all shape types once
    g_mesh_ids[SHAPE_ICOSAHEDRON] = atlas_add_icosahedron(atlas);
    g_mesh_ids[SHAPE_BOX]         = atlas_add_box(atlas);
    g_mesh_ids[SHAPE_TETRAHEDRON] = atlas_add_tetrahedron(atlas);
    g_mesh_ids[SHAPE_OCTAHEDRON]  = atlas_add_octahedron(atlas);

    int grid_size = (int)std::ceil(std::sqrt((double)num_objects));
    float spacing = 3.0f;
    float start_height = 10.0f;

    for (int i = 0; i < num_objects; i++) {
        int row = i / grid_size, col = i % grid_size;
        float x_off = -(grid_size - 1) * spacing * 0.5f;
        float z_off = -(grid_size - 1) * spacing * 0.5f;
        float noise = 0.2f;

        auto rn = [&]() { return ((float)rand() / RAND_MAX * 2.0f - 1.0f) * noise; };
        auto rf = [&](float lo, float hi) { return lo + (float)rand() / RAND_MAX * (hi - lo); };

        ShapeType shape = (ShapeType)(i % SHAPE_COUNT);
        float sx = rf(0.5f, 1.8f);
        float sy = rf(0.5f, 1.8f);
        float sz = rf(0.5f, 1.8f);

        int gjk_count = 0;
        float* gjk_verts = build_gjk_verts(shape, &gjk_count);

        float scale[3] = { sx, sy, sz };
        float br = compute_bounding_radius(gjk_verts, gjk_count, scale);

        objects[i].position[0] = x_off + col * spacing + rn();
        objects[i].position[1] = start_height + (i % 3) * 2.0f + rn();
        objects[i].position[2] = z_off + row * spacing + rn();

        objects[i].velocity[0] = ((i % 3) - 1) * 1.0f + rn();
        objects[i].velocity[1] = rn();
        objects[i].velocity[2] = ((i % 2) - 0.5f) * 2.0f + rn();

        objects[i].scale[0] = sx;
        objects[i].scale[1] = sy;
        objects[i].scale[2] = sz;

        memcpy(objects[i].color, colors[i % NUM_COLORS], sizeof(float) * 4);

        float mass = sx * sy * sz;
        objects[i].mass            = mass;
        objects[i].bounding_radius = br;
        objects[i].gjk_verts       = gjk_verts;
        objects[i].num_gjk_verts   = gjk_count;
        objects[i].mesh_id         = g_mesh_ids[shape];

        // Inverse principal inertia moments in body frame.
        // All four shapes have isotropic second moment: <x²> = <y²> = <z²> = k
        // With non-uniform scale: Ixx = mass*k*(sy²+sz²), etc.
        // k values (derived from polyhedral volume integrals):
        //   Box          (half-extents ±0.5):   k = 1/12
        //   Tetrahedron  (unit circumradius 1):  k = 1/15
        //   Octahedron   (unit circumradius 1):  k = 1/10
        //   Icosahedron  (unit circumradius 1):  k = (1 + 1/√5) / 10
        float k;
        switch (shape) {
            case SHAPE_BOX:         k = 1.0f / 12.0f; break;
            case SHAPE_TETRAHEDRON: k = 1.0f / 15.0f; break;
            case SHAPE_OCTAHEDRON:  k = 1.0f / 10.0f; break;
            case SHAPE_ICOSAHEDRON: k = (1.0f + 1.0f / std::sqrt(5.0f)) / 10.0f; break;
            default:                k = 1.0f / 10.0f; break;
        }
        float Ixx = mass * k * (sy*sy + sz*sz);
        float Iyy = mass * k * (sx*sx + sz*sz);
        float Izz = mass * k * (sx*sx + sy*sy);
        objects[i].inv_inertia[0] = 1.0f / Ixx;
        objects[i].inv_inertia[1] = 1.0f / Iyy;
        objects[i].inv_inertia[2] = 1.0f / Izz;
    }
}

static void free_scene_gjk_verts(ObjectInitData* objects, int num_objects) {
    for (int i = 0; i < num_objects; i++) {
        free(objects[i].gjk_verts);
        objects[i].gjk_verts = nullptr;
    }
}

int main(void) {
    int screenWidth  = 1200;
    int screenHeight = 800;

    if (!glfwInit()) { fprintf(stderr, "Failed to init GLFW\n"); return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    GLFWwindow* window = glfwCreateWindow(screenWidth, screenHeight,
                                          "Physics Simulation", NULL, NULL);
    if (!window) { fprintf(stderr, "Failed to create window\n"); glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        fprintf(stderr, "Failed to init GLAD\n"); return -1;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glClearColor(0.95f, 0.95f, 0.95f, 1.0f);

    input_init(window);

    Camera3D camera;
    float boundary = COMPUTE_BOUNDARY(NUM_OBJECTS);
    glm::vec3 cam_pos(0.0f, boundary * 0.8f, boundary * 1.6f);
    glm::vec3 cam_target(0.0f, -2.0f, 0.0f);
    camera_init(&camera, cam_pos, cam_target, 45.0f);

    // ---- Pre-init: build all CPU data before touching CUDA or GL resources ----
    std::srand((unsigned int)std::time(nullptr));
    MeshAtlas      atlas;   atlas_init(&atlas);
    ObjectInitData* objects = new ObjectInitData[NUM_OBJECTS];
    build_scene(&atlas, objects, NUM_OBJECTS);

    // ---- GL init ----

    // Create the dynamic position buffer (float4 per object) — registered with CUDA in sim_init
    GLuint gl_pos_buffer;
    glGenBuffers(1, &gl_pos_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, gl_pos_buffer);
    glBufferData(GL_ARRAY_BUFFER, NUM_OBJECTS * sizeof(float) * 4, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Create the dynamic quaternion buffer (float4 per object) — registered with CUDA in sim_init
    GLuint gl_quat_buffer;
    glGenBuffers(1, &gl_quat_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, gl_quat_buffer);
    glBufferData(GL_ARRAY_BUFFER, NUM_OBJECTS * sizeof(float) * 4, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    OpenGLRenderer renderer;
    if (!renderer_init(&renderer, &atlas, objects, NUM_OBJECTS, gl_pos_buffer, gl_quat_buffer)) {
        fprintf(stderr, "Failed to init renderer\n"); return -1;
    }
    atlas_free(&atlas);  // uploaded to GPU, no longer needed

    // ---- CUDA sim init ----
    if (!sim_init(objects, NUM_OBJECTS, gl_pos_buffer, gl_quat_buffer)) {
        fprintf(stderr, "Failed to init sim\n"); return -1;
    }
    free_scene_gjk_verts(objects, NUM_OBJECTS);
    delete[] objects;

    // ---- Physics params ----
    PhysicsParams params;
    params.gravity[0]         = 0.0f;
    params.gravity[1]         = GRAVITY_Y;
    params.gravity[2]         = 0.0f;
    params.delta_time         = DELTA_TIME;
    params.damping            = DAMPING_COEFF;
    params.boundary           = boundary;
    params.collision_epsilon  = COLLISION_EPSILON;

    double last_time         = glfwGetTime();
    double last_physics_time = last_time;
    double last_frame_time   = last_time;
    int    frame_count       = 0;
    int    fps               = 0;
    int    collision_count   = 0;
    double physics_accum     = 0.0;

    while (!glfwWindowShouldClose(window)) {
        frame_count++;
        double now       = glfwGetTime();
        float  deltaTime = (float)(now - last_frame_time);
        last_frame_time  = now;

        if (now - last_time >= 1.0) {
            fps = frame_count; frame_count = 0; last_time = now;
        }

        input_update(window);

        // Handle window resize
        glfwGetFramebufferSize(window, &screenWidth, &screenHeight);
        if (screenWidth > 0 && screenHeight > 0)
            glViewport(0, 0, screenWidth, screenHeight);

        if (IsKeyPressed(GLFW_KEY_ESCAPE)) glfwSetWindowShouldClose(window, GLFW_TRUE);
        if (IsKeyPressed(GLFW_KEY_R))      camera_reset(&camera, boundary);

        camera_update_controls(&camera, deltaTime);
        camera_update_matrices(&camera, screenHeight > 0 ? (float)screenWidth / screenHeight : 1.0f);

        // Physics: fixed timestep
        physics_accum += now - last_physics_time;
        last_physics_time = now;
        if (physics_accum > 0.2) physics_accum = 0.2;

        while (physics_accum >= DELTA_TIME) {
            collision_count = sim_broad_phase(&params);
            sim_step(&params);
            physics_accum -= DELTA_TIME;
        }

        // Copy positions from CUDA directly into GL buffer (no CPU readback)
        sim_copy_to_gl();

        renderer_draw(&renderer, camera.projection_matrix, camera.view_matrix);

        static int stats = 0;
        if (++stats % 60 == 0)
            printf("FPS: %d | Pairs: %d | Objects: %d\n", fps, collision_count, NUM_OBJECTS);

        glfwSwapBuffers(window);
    }

    renderer_cleanup(&renderer);
    glDeleteBuffers(1, &gl_pos_buffer);
    glDeleteBuffers(1, &gl_quat_buffer);
    sim_cleanup();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
