#version 460 core

// PBR surface shader, ported from CIS 5610 hw09 pbr.frag.glsl.
// Material uniforms became per-object flat varyings (SSBO-instanced MDI draw);
// IBL maps are baked at startup by ibl_precompute.cpp.

in vec3 v_world_normal;
in vec3 v_world_pos;
in vec2 v_uv;
flat in vec4 v_color;     // rgb = linear albedo (flat-color objects)
flat in float v_tex_index;
flat in vec2 v_pbr;       // x = metallic, y = roughness

uniform vec3 uCameraPos;
uniform sampler2DArray uTexArray;

// Image-based lighting
uniform samplerCube u_DiffuseIrradianceMap;
uniform samplerCube u_GlossyIrradianceMap;
uniform sampler2D u_BRDFLookupTexture;

out vec4 frag_color;

#define INV_PI 0.3183101550488765243077549
#define PI 3.14159265358979323846

float dotClamped(vec3 a, vec3 b) {
    return max(dot(a, b), 0.001f);
}

vec3 reinhard(vec3 col) {
    col = col / (col + vec3(1.f));
    return pow(col, vec3(1.f / 2.2f));
}

vec3 F(vec3 R, float cosTheta, float roughness) {
    float oneMCos = 1.f - cosTheta;
    float pow2 = oneMCos * oneMCos;
    float pow5 = pow2 * pow2 * oneMCos;
    vec3 t2 = max(vec3(1.f - roughness), R) - R;
    return R + t2 * pow5;
}

void main()
{
    vec3  N                = normalize(v_world_normal);
    vec3  albedo           = v_color.rgb;
    float metallic         = v_pbr.x;
    float roughness        = v_pbr.y;
    float ambientOcclusion = 1.0;

    // Texture array is GL_SRGB8_ALPHA8 — the GPU already decodes to linear,
    // so no pow(2.2) here (unlike hw09, which loads textures as raw RGBA).
    if (v_tex_index >= 0.0) {
        albedo = texture(uTexArray, vec3(v_uv, v_tex_index)).rgb;
    }

    vec3 irradiance = texture(u_DiffuseIrradianceMap, N).rgb;

    vec3 wo = normalize(uCameraPos - v_world_pos);
    vec3 wi = reflect(-wo, N);
    vec3 wh = normalize(wi + wo);

    vec3 R = mix(vec3(0.04), albedo, metallic);
    vec3 f  = F(R, max(dot(wh, wo), 0.f), roughness);

    vec3 kd = vec3(1.0f) - f;
    kd *= (1.0 - metallic);
    vec3 diffuse    = irradiance * albedo;

    const float MAX_REFLECTION_LOD = 4.0;
    vec3 prefilteredColor = textureLod(u_GlossyIrradianceMap, wi,  roughness * MAX_REFLECTION_LOD).rgb;
    vec2 dg  = texture(u_BRDFLookupTexture, vec2(dotClamped(N, wo), roughness)).rg;
    vec3 specular = prefilteredColor * (f * dg.x + dg.y);

    vec3 ambient    = (kd * diffuse + specular) * ambientOcclusion;

    vec3 Lo = ambient;
    Lo = reinhard(Lo);
    frag_color = vec4(Lo, 1.f);
}