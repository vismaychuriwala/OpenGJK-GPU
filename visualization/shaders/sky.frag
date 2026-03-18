#version 330 core

in vec3 vDir;
uniform sampler2D uEnvMap;
out vec4 FragColor;

vec2 sampleSphericalMap(vec3 v) {
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv *= vec2(0.1591, 0.3183);
    return uv + 0.5;
}

void main() {
    vec3 dir   = normalize(vDir);
    vec3 color = texture(uEnvMap, sampleSphericalMap(dir)).rgb;
    color = color / (color + vec3(1.0)); // Reinhard tone map
    FragColor = vec4(color, 1.0);
}
