#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <cmath>
#include <string>
#ifdef _WIN32
#include <windows.h>
#endif

// Returns the directory containing the running executable (with trailing slash).
static std::string exe_dir() {
#ifdef _WIN32
    char buf[MAX_PATH];
    GetModuleFileNameA(nullptr, buf, MAX_PATH);
    std::string s(buf);
    auto pos = s.find_last_of("\\/");
    return (pos == std::string::npos) ? "./" : s.substr(0, pos + 1);
#else
    return "./";
#endif
}

#include "rendering/input.h"
#include "rendering/camera.h"
#include "rendering/opengl_renderer.h"
#include "rendering/mesh_builder.h"
#include "sim_config.h"
#include "sim_api.h"

// Golden-ratio HSV: spreads hues maximally across all objects
static void hsv_to_rgb(float h, float s, float v, float* r, float* g, float* b) {
    int   hi = (int)(h * 6.0f) % 6;
    float f  = h * 6.0f - (int)(h * 6.0f);
    float p  = v * (1.0f - s);
    float q  = v * (1.0f - f * s);
    float t  = v * (1.0f - (1.0f - f) * s);
    switch (hi) {
        case 0: *r = v; *g = t; *b = p; break;
        case 1: *r = q; *g = v; *b = p; break;
        case 2: *r = p; *g = v; *b = t; break;
        case 3: *r = p; *g = q; *b = v; break;
        case 4: *r = t; *g = p; *b = v; break;
        default:*r = v; *g = p; *b = q; break;
    }
}

static void object_color(int i, float* rgba) {
    float hue = std::fmod(i * 0.618033988f, 1.0f);        // golden ratio
    float sat = 0.65f + std::fmod(i * 0.127f, 0.25f);     // 0.65–0.90
    float val = 0.75f + std::fmod(i * 0.211f, 0.20f);     // 0.75–0.95
    hsv_to_rgb(hue, sat, val, &rgba[0], &rgba[1], &rgba[2]);
    rgba[3] = 1.0f;
}

// Base shape types
enum ShapeType { SHAPE_ICOSAHEDRON = 0, SHAPE_BOX, SHAPE_TETRAHEDRON, SHAPE_OCTAHEDRON, SHAPE_BASE_COUNT };
static int g_mesh_ids[SHAPE_BASE_COUNT];

// Hull variants — generated at build_scene init
static int    g_num_hull_variants = 0;
static int*   g_hull_mesh_ids   = nullptr;
static float** g_hull_gjk_verts = nullptr;
static int*   g_hull_gjk_counts = nullptr;

#if LOAD_OBJS
// OBJ shapes — loaded from disk at startup.
// Add new paths here to include more meshes.
static const char* OBJ_PATHS[] = {
    "objs/wahoo.obj",
    "objs/alienanimal.obj",
};
static const int NUM_OBJ_SHAPES = sizeof(OBJ_PATHS) / sizeof(OBJ_PATHS[0]);
static int    g_obj_mesh_ids[NUM_OBJ_SHAPES];
static float* g_obj_gjk_verts[NUM_OBJ_SHAPES];
static int    g_obj_gjk_counts[NUM_OBJ_SHAPES];
static int    g_num_obj_loaded = 0;
#else
static const int g_num_obj_loaded = 0;
#endif

static float* build_gjk_verts(ShapeType type, int* out_count) {
    switch (type) {
        case SHAPE_ICOSAHEDRON: return gen_gjk_icosahedron(out_count);
        case SHAPE_BOX:         return gen_gjk_box(out_count);
        case SHAPE_TETRAHEDRON: return gen_gjk_tetrahedron(out_count);
        case SHAPE_OCTAHEDRON:  return gen_gjk_octahedron(out_count);
        default:                return gen_gjk_icosahedron(out_count);
    }
}

static void build_scene(MeshAtlas* atlas, ObjectInitData* objects, int num_objects) {
    // Register base shape types
    g_mesh_ids[SHAPE_ICOSAHEDRON] = atlas_add_icosahedron(atlas);
    g_mesh_ids[SHAPE_BOX]         = atlas_add_box(atlas);
    g_mesh_ids[SHAPE_TETRAHEDRON] = atlas_add_tetrahedron(atlas);
    g_mesh_ids[SHAPE_OCTAHEDRON]  = atlas_add_octahedron(atlas);

    // Generate unique random hull variants: HULL_SHAPE_RATIO * num_objects
    g_num_hull_variants = (int)(num_objects * HULL_SHAPE_RATIO);
    g_hull_mesh_ids   = (int*)   malloc(g_num_hull_variants * sizeof(int));
    g_hull_gjk_verts  = (float**)malloc(g_num_hull_variants * sizeof(float*));
    g_hull_gjk_counts = (int*)   malloc(g_num_hull_variants * sizeof(int));

    for (int h = 0; h < g_num_hull_variants; h++) {
        int n = MIN_HULL_VERTS + (h % (MAX_HULL_VERTS - MIN_HULL_VERTS + 1));
        int cnt = 0;
        float* pts = gen_gjk_random_hull(n, (unsigned int)(h * 2654435761u), &cnt);
        center_hull_verts(pts, cnt);          // shift so volumetric COM = origin
        g_hull_gjk_verts[h]  = pts;
        g_hull_gjk_counts[h] = cnt;
        g_hull_mesh_ids[h]   = atlas_add_convex_hull(atlas, pts, cnt);
    }

#if LOAD_OBJS
    g_num_obj_loaded = 0;
    std::string base = exe_dir();
    for (int o = 0; o < NUM_OBJ_SHAPES; o++) {
        std::string path = base + OBJ_PATHS[o];
        int ok = load_obj_shape(atlas, path.c_str(),
                                &g_obj_mesh_ids[o],
                                &g_obj_gjk_verts[o],
                                &g_obj_gjk_counts[o]);
        if (ok) { g_num_obj_loaded++; }
        else    { printf("Warning: could not load %s\n", path.c_str()); }
    }
#endif

    int grid_size = (int)std::ceil(std::sqrt((double)num_objects));
    float spacing = 3.0f;
    float start_height = 10.0f;

    int total_types = SHAPE_BASE_COUNT + g_num_hull_variants + g_num_obj_loaded;

    for (int i = 0; i < num_objects; i++) {
        int row = i / grid_size, col = i % grid_size;
        float x_off = -(grid_size - 1) * spacing * 0.5f;
        float z_off = -(grid_size - 1) * spacing * 0.5f;
        float noise = 0.2f;

        auto rn = [&]() { return ((float)rand() / RAND_MAX * 2.0f - 1.0f) * noise; };
        auto rf = [&](float lo, float hi) { return lo + (float)rand() / RAND_MAX * (hi - lo); };

        int type_idx = i % total_types;
        int    gjk_count = 0;
        float* gjk_verts = nullptr;
        int    mesh_id   = 0;

        if (type_idx < g_num_hull_variants) {
            // Random hull variant
            int hull_id = type_idx;
            gjk_count = g_hull_gjk_counts[hull_id];
            gjk_verts = (float*)malloc(gjk_count * 3 * sizeof(float));
            memcpy(gjk_verts, g_hull_gjk_verts[hull_id], gjk_count * 3 * sizeof(float));
            mesh_id = g_hull_mesh_ids[hull_id];
        }
#if LOAD_OBJS
        else if (type_idx < g_num_hull_variants + g_num_obj_loaded) {
            // OBJ shape
            int obj_id = type_idx - g_num_hull_variants;
            gjk_count = g_obj_gjk_counts[obj_id];
            gjk_verts = (float*)malloc(gjk_count * 3 * sizeof(float));
            memcpy(gjk_verts, g_obj_gjk_verts[obj_id], gjk_count * 3 * sizeof(float));
            mesh_id = g_obj_mesh_ids[obj_id];
        }
#endif
        else {
            // Base shape
            ShapeType shape = (ShapeType)((type_idx - g_num_hull_variants - g_num_obj_loaded) % SHAPE_BASE_COUNT);
            gjk_verts = build_gjk_verts(shape, &gjk_count);
            mesh_id   = g_mesh_ids[shape];
        }

        float base = rf(SCALE_BASE_MIN, SCALE_BASE_MAX);
        float sx = base * rf(SCALE_AXIS_MIN, SCALE_AXIS_MAX);
        float sy = base * rf(SCALE_AXIS_MIN, SCALE_AXIS_MAX);
        float sz = base * rf(SCALE_AXIS_MIN, SCALE_AXIS_MAX);

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

        object_color(i, objects[i].color);

        float mass = sx * sy * sz;
        objects[i].mass            = mass;
        objects[i].bounding_radius = br;
        objects[i].gjk_verts       = gjk_verts;
        objects[i].num_gjk_verts   = gjk_count;
        objects[i].mesh_id         = mesh_id;

        // Inertia: Ixx = mass*(ky*sy²+kz*sz²), etc.
        // Base shapes: analytic isotropic k (ky=kz=kx).
        // Random hulls: per-axis kx,ky,kz from polyhedral second-moment integral.
        float kx, ky, kz;
        bool is_base = (type_idx >= g_num_hull_variants + g_num_obj_loaded);
        if (is_base) {
            ShapeType shape = (ShapeType)((type_idx - g_num_hull_variants - g_num_obj_loaded) % SHAPE_BASE_COUNT);
            float k;
            switch (shape) {
                case SHAPE_BOX:         k = 1.0f / 12.0f; break;
                case SHAPE_TETRAHEDRON: k = 1.0f / 15.0f; break;
                case SHAPE_OCTAHEDRON:  k = 1.0f / 10.0f; break;
                default:                k = (1.0f + 1.0f / std::sqrt(5.0f)) / 10.0f; break;
            }
            kx = ky = kz = k;
        } else {
            compute_hull_inertia_k(gjk_verts, gjk_count, &kx, &ky, &kz);
        }
        float Ixx = mass * (ky*sy*sy + kz*sz*sz);
        float Iyy = mass * (kx*sx*sx + kz*sz*sz);
        float Izz = mass * (kx*sx*sx + ky*sy*sy);
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
    for (int h = 0; h < g_num_hull_variants; h++)
        free(g_hull_gjk_verts[h]);
    free(g_hull_mesh_ids);  free(g_hull_gjk_verts);  free(g_hull_gjk_counts);
    g_hull_mesh_ids = nullptr;  g_hull_gjk_verts = nullptr;  g_hull_gjk_counts = nullptr;
    g_num_hull_variants = 0;
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
    camera_init(&camera, glm::vec3(0.0f), glm::vec3(0.0f), 45.0f);
    camera_reset(&camera, boundary);

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
    params.gravity_y          = GRAVITY_Y;
    params.delta_time         = DELTA_TIME;
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
