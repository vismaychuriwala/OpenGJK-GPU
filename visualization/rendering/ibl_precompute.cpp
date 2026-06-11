// IBL pre-compute: equirect HDR → env cubemap → diffuse/glossy irradiance cubemaps.
// Ported from CIS 5610 hw09 (mygl.cpp renderCubeMapToTexture / renderConvolved*).
#include "ibl_precompute.h"
#include "opengl_renderer.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cstdio>
#include <cmath>

#define ENV_CUBE_DIM     1024
#define DIFFUSE_CUBE_DIM 32
#define GLOSSY_CUBE_DIM  512
#define GLOSSY_MIP_LEVELS 5

// -X, +X, -Y, +Y, -Z, +Z face view matrices (pure rotations; cube spans [-1,1]^3
// so view * pos lands directly in NDC — no projection needed)
static const glm::mat4 views[6] = {
    glm::lookAt(glm::vec3(0.0f), glm::vec3( 1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
    glm::lookAt(glm::vec3(0.0f), glm::vec3(-1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
    glm::lookAt(glm::vec3(0.0f), glm::vec3( 0.0f,  1.0f,  0.0f), glm::vec3(0.0f,  0.0f,  1.0f)),
    glm::lookAt(glm::vec3(0.0f), glm::vec3( 0.0f, -1.0f,  0.0f), glm::vec3(0.0f,  0.0f, -1.0f)),
    glm::lookAt(glm::vec3(0.0f), glm::vec3( 0.0f,  0.0f,  1.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
    glm::lookAt(glm::vec3(0.0f), glm::vec3( 0.0f,  0.0f, -1.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
};

// FBO + depth RBO + RGB16F cubemap render target
struct CubeMapFB {
    GLuint fbo;
    GLuint depth_rbo;
    GLuint cubemap;
    int    dim;
};

static void cubemap_fb_create(CubeMapFB* fb, int dim, bool mipmap) {
    fb->dim = dim;
    glGenFramebuffers(1, &fb->fbo);
    glGenRenderbuffers(1, &fb->depth_rbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fb->fbo);
    glBindRenderbuffer(GL_RENDERBUFFER, fb->depth_rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, dim, dim);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, fb->depth_rbo);
    GLenum draw_bufs[1] = { GL_COLOR_ATTACHMENT0 };
    glDrawBuffers(1, draw_bufs);

    glGenTextures(1, &fb->cubemap);
    glBindTexture(GL_TEXTURE_CUBE_MAP, fb->cubemap);
    for (unsigned int i = 0; i < 6; ++i)
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F,
                     dim, dim, 0, GL_RGB, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER,
                    mipmap ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    if (mipmap)
        glGenerateMipmap(GL_TEXTURE_CUBE_MAP);  // allocate the mip chain
}

// Releases the FBO/RBO but keeps the cubemap texture (the bake output).
static void cubemap_fb_release_fbo(CubeMapFB* fb) {
    glDeleteFramebuffers(1, &fb->fbo);
    glDeleteRenderbuffers(1, &fb->depth_rbo);
    fb->fbo = 0; fb->depth_rbo = 0;
}

// Renders all 6 faces of `fb` at `mip_level` with the bound program.
static void render_faces(const CubeMapFB* fb, GLint u_viewproj, int mip_level,
                         GLuint cube_vao) {
    for (int i = 0; i < 6; ++i) {
        glUniformMatrix4fv(u_viewproj, 1, GL_FALSE, &views[i][0][0]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                               fb->cubemap, mip_level);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glBindVertexArray(cube_vao);
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, nullptr);
    }
}

bool ibl_bake(GLuint equirect_tex, IBLMaps* out) {
    out->env_cubemap = out->diffuse_irradiance = out->glossy_prefilter = 0;

    // --- Bake shader programs ---
    char vert[1024], frag[1024];
    resolve_exe_relative("shaders/cubemap.vert", vert, sizeof(vert));

    resolve_exe_relative("shaders/cubemap_uv_conversion.frag", frag, sizeof(frag));
    GLuint prog_convert = create_shader_program(vert, frag);
    resolve_exe_relative("shaders/diffuseConvolution.frag", frag, sizeof(frag));
    GLuint prog_diffuse = create_shader_program(vert, frag);
    resolve_exe_relative("shaders/glossyConvolution.frag", frag, sizeof(frag));
    GLuint prog_glossy  = create_shader_program(vert, frag);
    if (!prog_convert || !prog_diffuse || !prog_glossy) {
        fprintf(stderr, "Failed to load IBL bake shaders\n");
        if (prog_convert) glDeleteProgram(prog_convert);
        if (prog_diffuse) glDeleteProgram(prog_diffuse);
        if (prog_glossy)  glDeleteProgram(prog_glossy);
        return false;
    }

    // --- Unit cube (positions + indices) ---
    static const float cube_pos[8][3] = {
        {-1,-1,-1}, {1,-1,-1}, {1,1,-1}, {-1,1,-1},
        {-1,-1, 1}, {1,-1, 1}, {1,1, 1}, {-1,1, 1},
    };
    static const GLuint cube_idx[36] = {
        1, 0, 3, 1, 3, 2,
        4, 5, 6, 4, 6, 7,
        5, 1, 2, 5, 2, 6,
        7, 6, 2, 7, 2, 3,
        0, 4, 7, 0, 7, 3,
        0, 1, 5, 0, 5, 4,
    };
    GLuint cube_vao, cube_vbo, cube_ebo;
    glGenVertexArrays(1, &cube_vao);
    glGenBuffers(1, &cube_vbo);
    glGenBuffers(1, &cube_ebo);
    glBindVertexArray(cube_vao);
    glBindBuffer(GL_ARRAY_BUFFER, cube_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cube_pos), cube_pos, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cube_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cube_idx), cube_idx, GL_STATIC_DRAW);
    glBindVertexArray(0);

    CubeMapFB env_fb, diffuse_fb, glossy_fb;
    cubemap_fb_create(&env_fb,     ENV_CUBE_DIM,     true);
    cubemap_fb_create(&diffuse_fb, DIFFUSE_CUBE_DIM, false);
    cubemap_fb_create(&glossy_fb,  GLOSSY_CUBE_DIM,  true);

    // --- 1. Equirectangular HDR → environment cubemap ---
    glUseProgram(prog_convert);
    glUniform1i(glGetUniformLocation(prog_convert, "u_EquirectangularMap"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, equirect_tex);
    glViewport(0, 0, ENV_CUBE_DIM, ENV_CUBE_DIM);
    glBindFramebuffer(GL_FRAMEBUFFER, env_fb.fbo);
    render_faces(&env_fb, glGetUniformLocation(prog_convert, "u_ViewProj"), 0, cube_vao);

    // --- 2. Mip the env cubemap (reduces fireflies in the glossy convolution) ---
    glBindTexture(GL_TEXTURE_CUBE_MAP, env_fb.cubemap);
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);

    // --- 3. Diffuse irradiance convolution ---
    glUseProgram(prog_diffuse);
    glUniform1i(glGetUniformLocation(prog_diffuse, "u_EnvironmentMap"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, env_fb.cubemap);
    glViewport(0, 0, DIFFUSE_CUBE_DIM, DIFFUSE_CUBE_DIM);
    glBindFramebuffer(GL_FRAMEBUFFER, diffuse_fb.fbo);
    render_faces(&diffuse_fb, glGetUniformLocation(prog_diffuse, "u_ViewProj"), 0, cube_vao);

    // --- 4. Glossy GGX prefilter, one mip per roughness level ---
    glUseProgram(prog_glossy);
    glUniform1i(glGetUniformLocation(prog_glossy, "u_EnvironmentMap"), 0);
    GLint u_roughness = glGetUniformLocation(prog_glossy, "u_Roughness");
    GLint u_viewproj  = glGetUniformLocation(prog_glossy, "u_ViewProj");
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, env_fb.cubemap);
    glBindFramebuffer(GL_FRAMEBUFFER, glossy_fb.fbo);
    for (int mip = 0; mip < GLOSSY_MIP_LEVELS; ++mip) {
        int mip_dim = (int)(GLOSSY_CUBE_DIM * std::pow(0.5, mip));
        glBindRenderbuffer(GL_RENDERBUFFER, glossy_fb.depth_rbo);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, mip_dim, mip_dim);
        glViewport(0, 0, mip_dim, mip_dim);
        glUniform1f(u_roughness, (float)mip / (float)(GLOSSY_MIP_LEVELS - 1));
        render_faces(&glossy_fb, u_viewproj, mip, cube_vao);
    }

    // --- Cleanup: keep only the three cubemap textures ---
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindVertexArray(0);
    cubemap_fb_release_fbo(&env_fb);
    cubemap_fb_release_fbo(&diffuse_fb);
    cubemap_fb_release_fbo(&glossy_fb);
    glDeleteVertexArrays(1, &cube_vao);
    glDeleteBuffers(1, &cube_vbo);
    glDeleteBuffers(1, &cube_ebo);
    glDeleteProgram(prog_convert);
    glDeleteProgram(prog_diffuse);
    glDeleteProgram(prog_glossy);

    out->env_cubemap        = env_fb.cubemap;
    out->diffuse_irradiance = diffuse_fb.cubemap;
    out->glossy_prefilter   = glossy_fb.cubemap;
    return true;
}

void ibl_free(IBLMaps* maps) {
    if (maps->env_cubemap)        glDeleteTextures(1, &maps->env_cubemap);
    if (maps->diffuse_irradiance) glDeleteTextures(1, &maps->diffuse_irradiance);
    if (maps->glossy_prefilter)   glDeleteTextures(1, &maps->glossy_prefilter);
    maps->env_cubemap = maps->diffuse_irradiance = maps->glossy_prefilter = 0;
}