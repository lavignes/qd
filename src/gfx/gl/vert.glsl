#version 410 core

const uint NUM_INST_COMPONENTS = 6;

uniform mat4 proj;
uniform mat4 view;

uniform uint store;
uniform sampler1DArray sbo;

layout (location = 0) in vec3 pos;
layout (location = 1) in float tx;
layout (location = 2) in vec3 norm;
layout (location = 3) in float ty;
layout (location = 4) in vec4 color;

flat out uint tex;
out vec2 tex_coord;
out vec4 vtx_color;

mat4 fetchModel(uint offset) {
    mat4 model;
    for (uint i = 0; i < 4; i++) {
        model[i] = texelFetch(sbo, ivec2(offset + i, store), 0);
    }
    return model;
}

vec4 fetchBlend(uint offset) {
    return texelFetch(sbo, ivec2(offset + 4, store), 0);
}

uint fetchTex(uint offset) {
    return uint(texelFetch(sbo, ivec2(offset + 5, store), 0)[0]);
}

void main() {
    uint offset = gl_InstanceID * NUM_INST_COMPONENTS;

    mat4 model = fetchModel(offset);
    vec4 blend = fetchBlend(offset);

    tex = fetchTex(offset);
    vtx_color = color * blend;
    tex_coord = vec2(tx, ty);

    gl_Position = proj * view * model * vec4(pos, 1.0);
}

