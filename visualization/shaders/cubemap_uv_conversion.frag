#version 330 core

in vec3 fs_Pos;
out vec4 out_Col;
uniform sampler2D u_EquirectangularMap;

//                              1/(2PI), 1/PI
const vec2 normalize_uv = vec2(0.1591, 0.3183);

vec2 sampleSphericalMap(vec3 v) {
    // U is in the range [-PI, PI], V is [-PI/2, PI/2]
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    // Convert UV to [-0.5, 0.5] in U&V
    uv *= normalize_uv;
    // Convert UV to [0, 1]
    uv += 0.5;
    return uv;
}

void main() {
    vec2 uv = sampleSphericalMap(normalize(fs_Pos));
    vec3 color = texture(u_EquirectangularMap, uv).rgb;

    out_Col = vec4(color, 1.0);
}