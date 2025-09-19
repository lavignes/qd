#version 410 core

uniform sampler2DArray tbo;

flat in uint tex;
in vec2 tex_coord;
in vec4 vtx_color;

out vec4 color;

void main() {
    color = texture(tbo, vec3(tex_coord, tex)) * vtx_color;
}

