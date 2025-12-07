#version 330 core

// Template vertex data (single icosahedron)
layout(location = 0) in vec3 aVertexPos;
layout(location = 1) in vec3 aVertexNormal;

// Per-instance data (from CUDA/OpenGL interop)
layout(location = 2) in vec3 aInstancePos;
layout(location = 3) in float aInstanceRadius;
layout(location = 4) in vec4 aInstanceColor;

// Uniforms
uniform mat4 uProjection;
uniform mat4 uView;

// Outputs to fragment shader
out vec3 vNormal;
out vec4 vColor;
out vec3 vWorldPos;

void main() {
    // Transform template vertex: scale by radius, translate by position
    vec3 worldPos = aInstancePos + aVertexPos * aInstanceRadius;

    gl_Position = uProjection * uView * vec4(worldPos, 1.0);

    // Pass data to fragment shader
    vNormal = aVertexNormal;  // Normal doesn't need scaling (uniform scaling)
    vColor = aInstanceColor;
    vWorldPos = worldPos;
}
