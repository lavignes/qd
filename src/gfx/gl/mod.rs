use std::{marker::PhantomData, mem, ops::Range, ptr};

use bytemuck::{NoUninit, Pod, Zeroable};
use gl::types::{GLenum, GLint, GLintptr, GLsizei, GLsizeiptr, GLuint};

use crate::{
    math::{Cross, Dot, IV2, Mat4, V3, V4, Xform3},
    mem::{Alloc, BitAlloc, BuddyAlloc, Handles},
};

use super::{Camera, Drawable, Settings, Target, Vtx};

const SBO_SIZE: usize = 512;
const SBO_DIM: usize = 2048;

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

        let vbo = Buf::new(gl::ARRAY_BUFFER, settings.vtx_buffer_size);
        let vbo_size = (settings.vtx_buffer_size * mem::size_of::<Vtx>()).next_power_of_two();
        log::debug!("VBO: {} MiB", vbo_size / 1024 / 1024);

        let ibo = Buf::new(gl::ELEMENT_ARRAY_BUFFER, settings.idx_buffer_size);
        let ibo_size = (settings.idx_buffer_size * mem::size_of::<u32>()).next_power_of_two();
        log::debug!("IBO: {} MiB", ibo_size / 1024 / 1024);

        let tbo = TexBuf::new(settings.tex_dim, settings.tex_count);
        let tbo_size =
            settings.tex_dim * settings.tex_dim * settings.tex_count * mem::size_of::<u32>();
        log::debug!(
            "TBO: {} textures ({} MiB)",
            settings.tex_count,
            tbo_size / 1024 / 1024
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
            gl::Uniform1i(utbo, 0);
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
    pub fn mesh_map<'a>(&'a mut self, hnd: u32) -> (BufMap<'a, Vtx>, BufMap<'a, u32>) {
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
    pub fn tex_map<'a>(&'a mut self, hnd: u32) -> TexMap<'a> {
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
    hnd: u32,
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

    fn find_mesh_batch(&mut self, hnd: &u32) -> &mut MeshBatch {
        match self.gl.mesh_batches.binary_search_by(|batch| {
            // TODO: also need to check if the store is full
            if !batch.insts.is_empty() {
                batch.hnd.cmp(hnd)
            } else {
                std::cmp::Ordering::Greater
            }
        }) {
            Ok(idx) => &mut self.gl.mesh_batches[idx],
            Err(idx) => {
                self.gl.mesh_batches.insert(
                    idx,
                    MeshBatch {
                        hnd: *hnd,
                        store: 0, // TODO: need to actually allocate different stores
                        insts: Vec::new(),
                    },
                );
                &mut self.gl.mesh_batches[idx]
            }
        }
    }

    pub fn draw<'b, I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (&'b Xform3, &'b Drawable)>,
    {
        for (world, draw) in iter.into_iter() {
            match draw {
                Drawable::None => {}
                Drawable::Mesh { hnd, tex, blend } => {
                    self.find_mesh_batch(hnd).insts.push(MeshInst {
                        world: Mat4::from(world),
                        blend: *blend,
                        tex: V4([*tex as f32, 0.0, 0.0, 0.0]),
                    });
                }
            }
        }
    }
}

impl<'a> Drop for Pass<'a> {
    fn drop(&mut self) {
        unsafe {
            gl::ActiveTexture(gl::TEXTURE1);
        }
        for batch in &mut self.gl.mesh_batches {
            if batch.insts.is_empty() {
                break;
            }
            const NUM_INST_COMPONENTS: usize = mem::size_of::<MeshInst>() / mem::size_of::<V4>();
            let mut err;
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
                err = gl::GetError();
            }
            if err != gl::NO_ERROR {
                crate::fatal!(
                    "Failed to transfer storage handle {} to storage: {err:X}",
                    batch.store
                );
            }
            let (_, ihnd) = self.gl.meshes[batch.hnd as usize];
            let range = &self.gl.ibo.inner.allocs.items[ihnd as usize].range;
            unsafe {
                gl::Uniform1ui(self.gl.ustore, batch.store);
                gl::DrawElementsInstancedBaseVertex(
                    gl::TRIANGLES,
                    (range.len() / mem::size_of::<u32>()) as GLsizei,
                    gl::UNSIGNED_INT,
                    ptr::without_provenance(range.start),
                    batch.insts.len() as GLsizei,
                    // we store index values relative to their offset in the index buffer
                    (range.start / mem::size_of::<u32>()) as GLint,
                );
                err = gl::GetError();
            }
            if err != gl::NO_ERROR {
                crate::fatal!("Failed to draw batch: {err:X}");
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
            inner: RawBuf::new(target, size * mem::size_of::<T>()),
            _marker: PhantomData,
        }
    }

    #[inline]
    fn alloc(&mut self, size: usize) -> u32 {
        self.inner.alloc(size * mem::size_of::<T>())
    }

    #[inline]
    fn map<'a>(&'a mut self, hnd: u32) -> BufMap<'a, T> {
        BufMap {
            inner: self.inner.map(hnd),
            _marker: PhantomData,
        }
    }
}

struct RawBuf {
    hnd: GLuint,
    target: GLenum,
    alloc: BuddyAlloc,
    allocs: Handles<Alloc>,
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
            alloc: BuddyAlloc::new(size, 512),
            allocs: Handles::new(),
        }
    }

    fn alloc(&mut self, size: usize) -> u32 {
        if let Some(alloc) = self.alloc.alloc(size) {
            return self.allocs.track(alloc) as u32;
        }
        crate::fatal!("Out of contiguous buffer space");
    }

    fn free(&mut self, hnd: u32) {
        self.allocs.untrack(hnd as usize);
    }

    #[inline]
    fn map<'a>(&'a mut self, hnd: u32) -> RawMap<'a> {
        let range = self.allocs.items[hnd as usize].range.clone();
        log::trace!(
            "Mapping buffer handle {hnd} ({}:{})",
            range.start,
            range.len()
        );
        RawMap {
            buf: self,
            hnd,
            range,
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
    range: Range<usize>,
}

impl<'a> RawMap<'a> {
    fn write(&mut self, data: &[u8]) {
        let err;
        unsafe {
            gl::BindBuffer(self.buf.target, self.buf.hnd);
            gl::BufferSubData(
                self.buf.target,
                self.range.start as GLintptr,
                self.range.len() as GLsizeiptr,
                data.as_ptr() as _,
            );
            err = gl::GetError();
        }
        if err != gl::NO_ERROR {
            crate::fatal!(
                "Failed to transfer buffer handle {} ({}:{}) to buffer: {err:X}",
                self.hnd,
                self.range.start,
                self.range.len()
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
            self.range.start,
            self.range.len()
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
    dim: usize,
    alloc: BitAlloc,
}

impl TexBuf {
    fn new(dim: usize, size: usize) -> Self {
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
            gl::ActiveTexture(gl::TEXTURE0);
            gl::BindTexture(gl::TEXTURE_2D_ARRAY, hnd);
            gl::TexStorage3D(
                gl::TEXTURE_2D_ARRAY,
                1,
                gl::RGBA8,
                dim as GLsizei,
                dim as GLsizei,
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
            dim,
            alloc: BitAlloc::new(size),
        }
    }

    fn alloc(&mut self) -> u32 {
        if let Some(hnd) = self.alloc.alloc() {
            return hnd as u32;
        }
        crate::fatal!("Out of texture space");
    }

    fn free(&mut self, hnd: u32) {
        self.alloc.free(hnd as usize);
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
            gl::ActiveTexture(gl::TEXTURE0);
            gl::TexSubImage3D(
                gl::TEXTURE_2D_ARRAY,
                0,
                0,
                0,
                self.hnd as GLint,
                self.buf.dim as GLsizei,
                self.buf.dim as GLsizei,
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
    alloc: BitAlloc,
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
            gl::ActiveTexture(gl::TEXTURE1);
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
            alloc: BitAlloc::new(size),
        }
    }

    fn alloc(&mut self) -> u32 {
        if let Some(hnd) = self.alloc.alloc() {
            return hnd as u32;
        }
        crate::fatal!("Out of storage space");
    }

    fn free(&mut self, hnd: u32) {
        self.alloc.free(hnd as usize);
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
    let vshader = compile_shader(gl::VERTEX_SHADER, include_str!("vert.glsl"));
    let fshader = compile_shader(gl::FRAGMENT_SHADER, include_str!("frag.glsl"));
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
