#version 330 core

// Template vertex data (single icosahedron)
layout(location = 0) in vec3 aVertexPos;

// Per-instance data
layout(location = 2) in vec3 aInstancePos;
layout(location = 3) in float aInstanceRadius;

// Uniforms
uniform mat4 uProjection;
uniform mat4 uView;

void main() {
    // Transform template vertex: scale by radius, translate by position
    vec3 worldPos = aInstancePos + aVertexPos * aInstanceRadius;
    gl_Position = uProjection * uView * vec4(worldPos, 1.0);
}
