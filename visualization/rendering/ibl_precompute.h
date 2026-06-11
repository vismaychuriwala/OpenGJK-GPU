#pragma once
#include <glad/glad.h>

// Baked image-based-lighting cubemaps (split-sum approximation).
// Produced once at startup by ibl_bake() from an equirectangular HDR texture.
typedef struct {
    GLuint env_cubemap;        // 1024^2 RGB16F, mipped — photographic background
    GLuint diffuse_irradiance; // 32^2  RGB16F — Lambertian convolution
    GLuint glossy_prefilter;   // 512^2 RGB16F, 5 mips — GGX prefilter (mip = roughness)
} IBLMaps;

// Renders the three cubemaps from `equirect_tex`. Requires a current GL context.
// Loads bake shaders from shaders/ next to the executable. Returns false on failure.
bool ibl_bake(GLuint equirect_tex, IBLMaps* out);
void ibl_free(IBLMaps* maps);