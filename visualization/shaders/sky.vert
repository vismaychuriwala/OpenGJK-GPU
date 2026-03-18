#version 330 core

uniform mat4 uInvProjView;
out vec3 vDir;

void main() {
    // Fullscreen triangle from vertex ID — no vertex buffer needed
    float x = float((gl_VertexID & 1) << 2) - 1.0;
    float y = float((gl_VertexID & 2) << 1) - 1.0;
    gl_Position = vec4(x, y, 1.0, 1.0);

    // Unproject to world-space direction (view matrix is rotation-only, no translation)
    vec4 dir = uInvProjView * vec4(x, y, 1.0, 1.0);
    vDir = dir.xyz / dir.w;
}
