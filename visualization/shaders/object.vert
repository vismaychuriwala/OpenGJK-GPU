#version 460 core

layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;

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

layout(std430, binding = 2) readonly buffer QuatSSBO {
    vec4 quats[];  // xyzw quaternion rotation
};

out vec3 v_world_normal;
out vec3 v_world_pos;
out vec4 v_color;
out vec2 v_uv;
flat out float v_tex_index;

vec3 quat_rotate(vec4 q, vec3 v) {
    vec3 u = q.xyz;
    float s = q.w;
    return 2.0 * dot(u, v) * u + (2.0*s*s - 1.0) * v + 2.0 * s * cross(u, v);
}

void main() {
    int id = gl_DrawID;

    vec3 scale     = objects[id].scale_pad.xyz;
    vec3 world_pos = positions[id].xyz;
    vec4 q         = quats[id];

    // Scale in local space, then rotate, then translate
    vec3 scaled  = a_pos * scale;
    vec3 rotated = quat_rotate(q, scaled);

    v_world_pos    = rotated + world_pos;
    v_world_normal = normalize(quat_rotate(q, a_normal / scale));
    v_color        = objects[id].color;
    v_uv           = a_uv;
    v_tex_index    = objects[id].scale_pad.w;

    gl_Position = uProjection * uView * vec4(v_world_pos, 1.0);
}
