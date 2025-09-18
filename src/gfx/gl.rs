use std::{marker::PhantomData, mem, ops::Range, ptr};

use bytemuck::{NoUninit, Pod, Zeroable};
use gl::types::{GLenum, GLint, GLintptr, GLsizei, GLsizeiptr, GLuint};

use crate::math::{Cross, Dot, IV2, Mat4, V3, V4, Xform3};

use super::{Camera, Drawable, Settings, Target, Vtx};

const VBO_SIZE: usize = 536870912;
const IBO_SIZE: usize = 536870912;
const TBO_SIZE: usize = 512;
const TEX_DIM: usize = 256;
const SBO_SIZE: usize = 512;
const SBO_DIM: usize = 1024;

pub struct Gl {
    vbo: Buf<Vtx>,
    ibo: Buf<u32>,
    tbo: TexBuf,
    sbo: StoreBuf,
    vao: GLuint,
    shader: GLuint,

    uproj: GLint,
    uview: GLint,
    utbo: GLint,
    usbo: GLint,
    ustore: GLint,

    meshes: Vec<(u32, u32)>,
    mesh_batches: Vec<MeshBatch>,
}

impl Gl {
    pub fn new(settings: &Settings) -> Self {
        log::trace!("Initializing Gfx...");
        let vbo = Buf::new(gl::ARRAY_BUFFER, VBO_SIZE);
        log::debug!("VBO: {} MiB", VBO_SIZE / 1024 / 1024);
        let ibo = Buf::new(gl::ELEMENT_ARRAY_BUFFER, IBO_SIZE);
        log::debug!("IBO: {} MiB", IBO_SIZE / 1024 / 1024);
        let tbo = TexBuf::new(TBO_SIZE);
        log::debug!(
            "TBO: {TBO_SIZE} textures ({} MiB)",
            (TEX_DIM * TEX_DIM * mem::size_of::<u32>() * TBO_SIZE) / 1024 / 1024
        );
        let sbo = StoreBuf::new(SBO_SIZE);
        log::debug!(
            "SBO: {SBO_SIZE} stores ({} MiB)",
            (SBO_DIM * mem::size_of::<V4>() * SBO_SIZE) / 1024 / 1024
        );

        let vao = create_vao();
        let shader = compile_and_link_shaders();

        let uproj;
        let uview;
        let utbo;
        let usbo;
        let ustore;
        unsafe {
            uproj = gl::GetUniformLocation(shader, c"proj".as_ptr());
            uview = gl::GetUniformLocation(shader, c"view".as_ptr());
            utbo = gl::GetUniformLocation(shader, c"tbo".as_ptr());
            usbo = gl::GetUniformLocation(shader, c"sbo".as_ptr());
            ustore = gl::GetUniformLocation(shader, c"store".as_ptr());
        }
        if uproj < 0 {
            crate::fatal!("Failed to locate 'proj' uniform in shader");
        }
        if uview < 0 {
            crate::fatal!("Failed to locate 'view' uniform in shader");
        }
        if utbo < 0 {
            crate::fatal!("Failed to locate 'tbo' uniform in shader");
        }
        if usbo < 0 {
            crate::fatal!("Failed to locate 'sbo' uniform in shader");
        }
        if ustore < 0 {
            crate::fatal!("Failed to locate 'store' uniform in shader");
        }

        unsafe {
            gl::UseProgram(shader);
            gl::BindVertexArray(vao);
            gl::BindBuffer(gl::ARRAY_BUFFER, vbo.inner.hnd);
            gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ibo.inner.hnd);
            gl::ActiveTexture(gl::TEXTURE0);
            gl::BindTexture(gl::TEXTURE_2D_ARRAY, tbo.hnd);
            gl::Uniform1i(utbo, 0);
            gl::ActiveTexture(gl::TEXTURE1);
            gl::BindTexture(gl::TEXTURE_1D_ARRAY, sbo.hnd);
            gl::Uniform1i(usbo, 1);

            let IV2([w, h]) = settings.size.into();
            gl::Viewport(0, 0, w, h);
            gl::Enable(gl::DEPTH_TEST);
            gl::Enable(gl::CULL_FACE);
            gl::CullFace(gl::BACK);
        }

        log::trace!("Initialized Gfx");
        Self {
            vbo,
            ibo,
            tbo,
            sbo,
            vao,
            shader,

            uproj,
            uview,
            utbo,
            usbo,
            ustore,

            meshes: Vec::new(),
            mesh_batches: Vec::new(),
        }
    }

    #[inline]
    pub fn pass<'a>(&'a mut self, target: Target, camera: &'a Camera) -> Pass<'a> {
        let view = look_at(camera.pos, camera.at, V3::UP);
        unsafe {
            gl::UniformMatrix4fv(
                self.uproj,
                1,
                gl::FALSE,
                Mat4::from(camera.proj).0.as_ptr() as _,
            );
            gl::UniformMatrix4fv(self.uview, 1, gl::FALSE, view.0.as_ptr() as _);
        }
        Pass {
            gl: self,
            target,
            camera,
        }
    }

    #[inline]
    pub fn mesh_alloc(&mut self, vtxs: usize, idxs: usize) -> u32 {
        let vhnd = self.vbo.alloc(vtxs);
        let ihnd = self.ibo.alloc(idxs);
        let hnd = self.meshes.len();
        self.meshes.push((vhnd, ihnd));
        hnd as u32
    }

    #[inline]
    pub fn mesh_map(&mut self, hnd: u32) -> (BufMap<'_, Vtx>, BufMap<'_, u32>) {
        let &mut Self {
            ref mut vbo,
            ref mut ibo,
            ref meshes,
            ..
        } = self;
        let (vhnd, ihnd) = meshes[hnd as usize];
        (vbo.map(vhnd), ibo.map(ihnd))
    }

    #[inline]
    pub fn tex_alloc(&mut self) -> u32 {
        self.tbo.alloc()
    }

    #[inline]
    pub fn tex_map(&mut self, hnd: u32) -> TexMap<'_> {
        log::trace!("Mapping texture handle {hnd}",);
        TexMap {
            buf: &mut self.tbo,
            hnd,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MeshInst {
    world: Mat4,
    blend: V4,
    tex: V4,
}

struct MeshBatch {
    range: Range<usize>,
    store: u32,
    insts: Vec<MeshInst>,
}

pub struct Pass<'a> {
    gl: &'a mut Gl,
    target: Target,
    camera: &'a Camera,
}

impl<'a> Pass<'a> {
    #[inline]
    pub fn clear_all(&mut self) {
        unsafe {
            gl::ClearColor(0.0, 0.0, 0.0, 0.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
        }
    }

    pub fn draw<'b, I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (&'b Xform3, &'b Drawable)>,
    {
        // TODO dont clear. append to existing batches (each will be empty)
        self.gl.mesh_batches.clear();

        for (world, draw) in iter.into_iter() {
            match draw {
                Drawable::None => {}
                Drawable::Mesh { hnd, tex, blend } => {
                    let (_, ihnd) = self.gl.meshes[*hnd as usize];
                    let range = &self.gl.ibo.inner.used[ihnd as usize];

                    self.gl.mesh_batches.push(MeshBatch {
                        range: range.clone(),
                        store: 0, // TODO: might overflow the store
                        insts: vec![MeshInst {
                            world: Mat4::from(world),
                            blend: *blend,
                            tex: V4([*tex as f32, 0.0, 0.0, 0.0]),
                        }],
                    });
                }
            }
        }
    }
}

impl<'a> Drop for Pass<'a> {
    fn drop(&mut self) {
        unsafe {
            gl::BindTexture(gl::TEXTURE_1D_ARRAY, self.gl.sbo.hnd);
        }
        for batch in &mut self.gl.mesh_batches {
            if batch.insts.is_empty() {
                break;
            }
            const NUM_INST_COMPONENTS: usize = mem::size_of::<MeshInst>() / mem::size_of::<V4>();
            unsafe {
                gl::TexSubImage2D(
                    gl::TEXTURE_1D_ARRAY,
                    0,
                    0,
                    batch.store as GLint,
                    (batch.insts.len() * NUM_INST_COMPONENTS) as GLsizei,
                    1,
                    gl::RGBA,
                    gl::FLOAT,
                    batch.insts.as_ptr() as _,
                );
                gl::Uniform1ui(self.gl.ustore, batch.store);
                gl::DrawElementsInstancedBaseVertex(
                    gl::TRIANGLES,
                    (batch.range.len() / mem::size_of::<u32>()) as GLsizei,
                    gl::UNSIGNED_INT,
                    ptr::without_provenance(batch.range.start),
                    batch.insts.len() as GLsizei,
                    // we store index values relative to their offset in the index buffer
                    (batch.range.start / mem::size_of::<u32>()) as GLint,
                );
            }
            batch.insts.clear();
        }
    }
}

fn look_at(pos: V3, at: V3, up: V3) -> Mat4 {
    let forward = (at - pos).normalized();
    let backward = -forward;
    let right = forward.cross(up).normalized();
    let up = right.cross(forward);
    Mat4([
        V4([right.0[0], up.0[0], backward.0[0], 0.0]),
        V4([right.0[1], up.0[1], backward.0[1], 0.0]),
        V4([right.0[2], up.0[2], backward.0[2], 0.0]),
        V4([-right.dot(pos), -up.dot(pos), forward.dot(pos), 1.0]),
    ])
}

struct Buf<T> {
    inner: RawBuf,
    _marker: PhantomData<T>,
}

impl<T> Buf<T> {
    #[inline]
    fn new(target: GLenum, size: usize) -> Self {
        Self {
            inner: RawBuf::new(target, size),
            _marker: PhantomData,
        }
    }

    #[inline]
    fn alloc(&mut self, size: usize) -> u32 {
        self.inner.alloc(size * mem::size_of::<T>())
    }

    #[inline]
    fn map(&mut self, hnd: u32) -> BufMap<'_, T> {
        BufMap {
            inner: self.inner.map(hnd),
            _marker: PhantomData,
        }
    }
}

struct RawBuf {
    hnd: GLuint,
    target: GLenum,
    len: usize,
    cap: usize,
    used: Vec<Range<usize>>,
    free: Vec<Range<usize>>,
}

impl RawBuf {
    fn new(target: GLenum, size: usize) -> Self {
        let mut hnd = 0;
        let mut err;
        unsafe {
            gl::GenBuffers(1, &mut hnd);
            err = gl::GetError();
        }
        if err != gl::NO_ERROR {
            crate::fatal!("Failed to name buffer: {err:X}");
        }
        unsafe {
            gl::BindBuffer(target, hnd);
            gl::BufferData(target, size as GLsizeiptr, ptr::null(), gl::DYNAMIC_DRAW);
            err = gl::GetError();
        }
        if err != gl::NO_ERROR {
            crate::fatal!("Failed to allocate buffer: {err:X}");
        }
        Self {
            hnd,
            target,
            len: 0,
            cap: size,
            used: Vec::new(),
            free: vec![0..size],
        }
    }

    fn alloc(&mut self, size: usize) -> u32 {
        for alloc in self.free.iter_mut() {
            if alloc.len() >= size {
                let hnd = self.used.len();
                self.used.push(alloc.start..(alloc.start + size));
                alloc.start += size;
                return hnd as u32;
            }
        }
        crate::fatal!("Out of contiguous buffer space");
    }

    #[inline]
    fn map(&mut self, hnd: u32) -> RawMap<'_> {
        let alloc = self.used[hnd as usize].clone();
        log::trace!(
            "Mapping buffer handle {hnd} ({}:{})",
            alloc.start,
            alloc.len()
        );
        RawMap {
            buf: self,
            hnd,
            alloc,
        }
    }
}

impl Drop for RawBuf {
    #[inline]
    fn drop(&mut self) {
        let err;
        unsafe {
            gl::DeleteBuffers(1, &self.hnd);
            err = gl::GetError();
        }
        if err != gl::NO_ERROR {
            crate::fatal!("Failed to free buffer: {err:X}");
        }
    }
}

struct RawMap<'a> {
    buf: &'a mut RawBuf,
    hnd: u32,
    alloc: Range<usize>,
}

impl<'a> RawMap<'a> {
    fn write(&mut self, data: &[u8]) {
        let err;
        unsafe {
            gl::BufferSubData(
                self.buf.target,
                self.alloc.start as GLintptr,
                self.alloc.len() as GLsizeiptr,
                data.as_ptr() as _,
            );
            err = gl::GetError();
        }
        if err != gl::NO_ERROR {
            crate::fatal!(
                "Failed to transfer buffer handle {} ({}:{}) to buffer: {err:X}",
                self.hnd,
                self.alloc.start,
                self.alloc.len()
            );
        }
    }
}

impl<'a> Drop for RawMap<'a> {
    #[inline]
    fn drop(&mut self) {
        log::trace!(
            "Unmapping buffer handle {} ({}:{})",
            self.hnd,
            self.alloc.start,
            self.alloc.len()
        );
    }
}

pub struct BufMap<'a, T> {
    inner: RawMap<'a>,
    _marker: PhantomData<T>,
}

impl<'a, T> BufMap<'a, T> {
    #[inline]
    pub fn write(&mut self, data: &[T])
    where
        T: NoUninit,
    {
        self.inner.write(bytemuck::cast_slice(data));
    }
}

struct TexBuf {
    hnd: GLuint,
    len: usize,
    cap: usize,
    used: Vec<Range<usize>>,
    free: Vec<Range<usize>>,
}

impl TexBuf {
    fn new(size: usize) -> Self {
        let mut hnd = 0;
        let mut err;
        unsafe {
            gl::GenTextures(1, &mut hnd);
            err = gl::GetError();
        }
        if err != gl::NO_ERROR {
            crate::fatal!("Failed to name texture: {err:X}");
        }
        unsafe {
            gl::BindTexture(gl::TEXTURE_2D_ARRAY, hnd);
            gl::TexStorage3D(
                gl::TEXTURE_2D_ARRAY,
                1,
                gl::RGBA8,
                TEX_DIM as GLsizei,
                TEX_DIM as GLsizei,
                size as GLsizei,
            );
            err = gl::GetError();
        }
        if err != gl::NO_ERROR {
            crate::fatal!("Failed to allocate texture: {err:X}");
        }
        unsafe {
            gl::TexParameteri(
                gl::TEXTURE_2D_ARRAY,
                gl::TEXTURE_MIN_FILTER,
                gl::LINEAR as GLint,
            );
            gl::TexParameteri(
                gl::TEXTURE_2D_ARRAY,
                gl::TEXTURE_MAG_FILTER,
                gl::LINEAR as GLint,
            );
            gl::TexParameteri(
                gl::TEXTURE_2D_ARRAY,
                gl::TEXTURE_WRAP_S,
                gl::CLAMP_TO_EDGE as GLint,
            );
            gl::TexParameteri(
                gl::TEXTURE_2D_ARRAY,
                gl::TEXTURE_WRAP_T,
                gl::CLAMP_TO_EDGE as GLint,
            );
            err = gl::GetError();
        }
        if err != gl::NO_ERROR {
            crate::fatal!("Failed to set texture parameters: {err:X}");
        }
        Self {
            hnd,
            len: 0,
            cap: size,
            used: Vec::new(),
            free: vec![0..size],
        }
    }

    fn alloc(&mut self) -> u32 {
        for alloc in self.free.iter_mut() {
            if alloc.len() > 0 {
                let hnd = self.used.len();
                self.used.push(alloc.start..(alloc.start + 1));
                alloc.start += 1;
                return hnd as u32;
            }
        }
        crate::fatal!("Out of texture space");
    }
}

impl Drop for TexBuf {
    #[inline]
    fn drop(&mut self) {
        let err;
        unsafe {
            gl::DeleteTextures(1, &self.hnd);
            err = gl::GetError();
        }
        if err != gl::NO_ERROR {
            crate::fatal!("Failed to free texture: {err:X}");
        }
    }
}

pub struct TexMap<'a> {
    buf: &'a mut TexBuf,
    hnd: u32,
}

impl<'a> TexMap<'a> {
    pub fn write(&mut self, data: &[u32]) {
        let err;
        unsafe {
            gl::BindTexture(gl::TEXTURE_2D_ARRAY, self.buf.hnd);
            gl::TexSubImage3D(
                gl::TEXTURE_2D_ARRAY,
                0,
                0,
                0,
                self.hnd as GLint,
                TEX_DIM as GLsizei,
                TEX_DIM as GLsizei,
                1,
                gl::RGBA,
                gl::UNSIGNED_BYTE,
                data.as_ptr() as _,
            );
            err = gl::GetError();
        }
        if err != gl::NO_ERROR {
            crate::fatal!(
                "Failed to transfer texture handle {} to texture: {err:X}",
                self.hnd
            );
        }
    }
}

impl<'a> Drop for TexMap<'a> {
    #[inline]
    fn drop(&mut self) {
        log::trace!("Unmapping texture handle {}", self.hnd);
    }
}

struct StoreBuf {
    hnd: GLuint,
    len: usize,
    cap: usize,
    used: Vec<Range<usize>>,
    free: Vec<Range<usize>>,
}

impl StoreBuf {
    fn new(size: usize) -> Self {
        let mut hnd = 0;
        let mut err;
        unsafe {
            gl::GenTextures(1, &mut hnd);
            err = gl::GetError();
        }
        if err != gl::NO_ERROR {
            crate::fatal!("Failed to name storage: {err:X}");
        }
        unsafe {
            gl::BindTexture(gl::TEXTURE_1D_ARRAY, hnd);
            gl::TexStorage2D(
                gl::TEXTURE_1D_ARRAY,
                1,
                gl::RGBA32F, // 1 `V4` per texel
                SBO_DIM as GLsizei,
                size as GLsizei,
            );
            err = gl::GetError();
        }
        if err != gl::NO_ERROR {
            crate::fatal!("Failed to allocate storage: {err:X}");
        }
        Self {
            hnd,
            len: 0,
            cap: size,
            used: Vec::new(),
            free: vec![0..size],
        }
    }

    fn alloc(&mut self) -> u32 {
        for alloc in self.free.iter_mut() {
            if alloc.len() > 0 {
                let hnd = self.used.len();
                self.used.push(alloc.start..(alloc.start + 1));
                alloc.start += 1;
                return hnd as u32;
            }
        }
        crate::fatal!("Out of storage space");
    }
}

impl Drop for StoreBuf {
    #[inline]
    fn drop(&mut self) {
        let err;
        unsafe {
            gl::DeleteTextures(1, &self.hnd);
            err = gl::GetError();
        }
        if err != gl::NO_ERROR {
            crate::fatal!("Failed to free storage: {err:X}");
        }
    }
}

fn create_vao() -> GLuint {
    let mut hnd = 0;
    let mut err;
    unsafe {
        gl::GenVertexArrays(1, &mut hnd);
        err = gl::GetError();
    }
    if err != gl::NO_ERROR {
        crate::fatal!("Failed to name attribute array: {err:X}");
    }
    const STRIDE: GLsizei = mem::size_of::<Vtx>() as GLsizei;
    unsafe {
        let pos = ptr::without_provenance(bytemuck::offset_of!(Vtx, pos));
        let tx = ptr::without_provenance(bytemuck::offset_of!(Vtx, tx));
        let norm = ptr::without_provenance(bytemuck::offset_of!(Vtx, norm));
        let ty = ptr::without_provenance(bytemuck::offset_of!(Vtx, ty));
        let color = ptr::without_provenance(bytemuck::offset_of!(Vtx, color));
        gl::BindVertexArray(hnd);
        gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, STRIDE, pos);
        gl::EnableVertexAttribArray(0);
        gl::VertexAttribPointer(1, 1, gl::FLOAT, gl::FALSE, STRIDE, tx);
        gl::EnableVertexAttribArray(1);
        gl::VertexAttribPointer(2, 3, gl::FLOAT, gl::FALSE, STRIDE, norm);
        gl::EnableVertexAttribArray(2);
        gl::VertexAttribPointer(3, 1, gl::FLOAT, gl::FALSE, STRIDE, ty);
        gl::EnableVertexAttribArray(3);
        gl::VertexAttribPointer(4, 4, gl::FLOAT, gl::FALSE, STRIDE, color);
        gl::EnableVertexAttribArray(4);
        err = gl::GetError();
    }
    if err != gl::NO_ERROR {
        crate::fatal!("Failed to configure attribute array: {err:X}");
    }
    hnd
}

const VSHADER: &'static str = r#"
    #version 410 core

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

    mat4 fetchModel() {
        mat4 model;
        for (uint i = 0; i < 4; i++) {
            model[i] = texelFetch(sbo, ivec2(gl_InstanceID + i, store), 0);
        }
        return model;
    }

    vec4 fetchBlend() {
        return texelFetch(sbo, ivec2(gl_InstanceID + 4, store), 0);
    }

    uint fetchTex() {
        return uint(texelFetch(sbo, ivec2(gl_InstanceID + 5, store), 0)[0]);
    }

    void main() {
        mat4 model = fetchModel();
        vec4 blend = fetchBlend();

        tex = fetchTex();
        vtx_color = color * blend;
        tex_coord = vec2(tx, ty);

        gl_Position = proj * view * model * vec4(pos, 1.0);
    }
"#;

const FSHADER: &'static str = r#"
    #version 410 core

    uniform sampler2DArray tbo;

    flat in uint tex;
    in vec2 tex_coord;
    in vec4 vtx_color;

    out vec4 color;

    void main() {
        color = texture(tbo, vec3(tex_coord, tex)) * vtx_color;
    }
"#;

fn compile_shader(kind: GLenum, src: &str) -> GLuint {
    let hnd;
    let err;
    unsafe {
        hnd = gl::CreateShader(kind);
        err = gl::GetError();
    }
    if err != gl::NO_ERROR {
        crate::fatal!("Failed to name shader: {err:X}");
    }
    let mut success: GLint = 0;
    unsafe {
        let len = src.len() as GLint;
        gl::ShaderSource(hnd, 1, mem::transmute(&src.as_ptr()), &len);
        gl::CompileShader(hnd);
        gl::GetShaderiv(hnd, gl::COMPILE_STATUS, &mut success);
    }
    if success == 0 {
        let mut len: GLint = 0;
        unsafe {
            gl::GetShaderiv(hnd, gl::INFO_LOG_LENGTH, &mut len);
        }
        let mut buf = " ".repeat(len as usize);
        unsafe {
            gl::GetShaderInfoLog(hnd, len, ptr::null_mut(), buf.as_mut_ptr() as _);
        }
        crate::fatal!("Failed to compile shader: {buf}");
    }
    hnd
}

fn attach_shader(program: GLuint, shader: GLuint) {
    let err;
    unsafe {
        gl::AttachShader(program, shader);
        err = gl::GetError();
    }
    if err != gl::NO_ERROR {
        crate::fatal!("Failed to attach shader: {err:X}");
    }
}

fn compile_and_link_shaders() -> GLuint {
    let vshader = compile_shader(gl::VERTEX_SHADER, VSHADER);
    let fshader = compile_shader(gl::FRAGMENT_SHADER, FSHADER);
    let hnd;
    let err;
    unsafe {
        hnd = gl::CreateProgram();
    }
    if hnd == 0 {
        unsafe {
            err = gl::GetError();
        }
        crate::fatal!("Failed to name program: {err:X}");
    }
    attach_shader(hnd, vshader);
    attach_shader(hnd, fshader);
    let mut success: GLint = 0;
    unsafe {
        gl::LinkProgram(hnd);
        gl::GetProgramiv(hnd, gl::LINK_STATUS, &mut success);
    }
    if success == 0 {
        let mut len: GLint = 0;
        unsafe {
            gl::GetProgramiv(hnd, gl::INFO_LOG_LENGTH, &mut len);
        }
        let mut buf = " ".repeat(len as usize);
        unsafe {
            gl::GetProgramInfoLog(hnd, len, ptr::null_mut(), buf.as_mut_ptr() as _);
        }
        crate::fatal!("Failed to link program: {buf}");
    }
    unsafe {
        gl::DeleteShader(vshader);
        gl::DeleteShader(fshader);
    }
    hnd
}
