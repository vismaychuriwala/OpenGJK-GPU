#version 460 core

layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;

uniform mat4 uProjection;
uniform mat4 uView;

struct ObjectStatic {
    vec4 scale_pad;  // xyz = scale, w = unused
    vec4 color;
};

layout(std430, binding = 0) readonly buffer StaticSSBO {
    ObjectStatic objects[];
};

layout(std430, binding = 1) readonly buffer DynamicSSBO {
    vec4 positions[];  // xyz = world pos, w = bounding_radius (unused here)
};

out vec3 v_world_normal;
out vec3 v_world_pos;
out vec4 v_color;

void main() {
    int id = gl_DrawID;

    vec3 scale    = objects[id].scale_pad.xyz;
    vec3 world_pos = positions[id].xyz;

    // Scale vertex, then translate to world space
    // No rotation yet — will apply quaternion here when EPA is integrated
    vec3 scaled = a_pos * scale;

    v_world_pos    = scaled + world_pos;
    v_world_normal = normalize(a_normal / scale);  // inverse-transpose for non-uniform scale
    v_color        = objects[id].color;

    gl_Position = uProjection * uView * vec4(v_world_pos, 1.0);
}
