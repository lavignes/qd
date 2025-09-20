use std::time::{Duration, Instant};

use qd::{
    gfx::{Camera, Drawable, Gfx, PassSettings, Proj, Settings, Target, Vtx},
    math::{UV2, V3, V4, Xform3},
    scene::{Node, Scene},
};
use sdl2::{
    event::Event,
    video::{GLProfile, SwapInterval},
};

fn main() {
    qd::log::init();

    let sdl = qd::ensure!(sdl2::init(), "Failed to initialize SDL: {}");
    let video = qd::ensure!(sdl.video(), "Failed to initialize SDL video: {}");

    // these will panic if the attributes fail to set
    let gl_attr = video.gl_attr();
    gl_attr.set_context_version(4, 1);
    gl_attr.set_context_profile(GLProfile::Core);

    let win = qd::ensure!(
        video
            .window("qd-sdl2", 1920, 1080)
            .opengl()
            .allow_highdpi()
            .position_centered()
            .build(),
        "Failed to create window: {}"
    );

    let gl_ctx = qd::ensure!(win.gl_create_context(), "Failed to create GL context: {}");
    qd::ensure!(win.gl_make_current(&gl_ctx));
    qd::ensure!(video.gl_set_swap_interval(SwapInterval::VSync));

    gl::load_with(|proc| video.gl_get_proc_address(proc) as *const _);

    let mut gfx = Gfx::new(&Settings {
        screen_size: UV2([1920, 1080]),

        vtx_buffer_size: 1024 * 1024 * 4,
        idx_buffer_size: 1024 * 1024 * 16,
        tex_dim: 256,
        tex_count: 512,
    });

    let mesh = gfx.mesh_alloc(4, 6);
    {
        let (mut vmap, mut imap) = gfx.mesh_map(mesh);
        vmap.write(&[
            Vtx {
                pos: V3([0.0, 0.0, 0.0]),
                color: V4::splat(1.0),
                ..Default::default()
            },
            Vtx {
                pos: V3([0.0, 32.0, 0.0]),
                color: V4::splat(1.0),
                ..Default::default()
            },
            Vtx {
                pos: V3([32.0, 0.0, 0.0]),
                color: V4::splat(1.0),
                ..Default::default()
            },
            Vtx {
                pos: V3([32.0, 32.0, 0.0]),
                color: V4::splat(1.0),
                ..Default::default()
            },
        ]);
        imap.write(&[0, 1, 2, 2, 1, 3]);
    }

    let tex = gfx.tex_alloc();
    {
        let mut tmap = gfx.tex_map(tex);
        tmap.write(&vec![0xFFFF00FF; 256 * 266]);
    }

    let mut events = qd::ensure!(sdl.event_pump());

    let mut frames = 0.0;
    let mut last = Instant::now();

    let mut scene = Scene::new();

    scene.add_node(Node {
        kid: 1,
        sib: Node::NONE,
        local: Xform3::IDENTITY,
        world: Xform3::IDENTITY,
        draw: Drawable::None,
    });

    const N: usize = 50;
    for i in 0..N {
        let mut local = Xform3::IDENTITY;
        local.scale.0[1] = 32.0;
        local.pos.0[0] = 32.0 * (i as f32);
        scene.add_node(Node {
            kid: Node::NONE,
            sib: if i < (N - 1) {
                (i + 2) as u32
            } else {
                Node::NONE
            },
            local,
            world: Xform3::IDENTITY,
            draw: Drawable::Mesh {
                hnd: mesh,
                tex,
                blend: V4::splat(((N - i) as f32) / (N as f32)),
            },
        });
    }

    let camera = Camera {
        pos: V3([-16.0, -16.0, 1.0]),
        at: V3([-16.0, -16.0, 0.0]),
        proj: Proj::Ortho {
            size: UV2([1920, 1080]),
            near: 0.0,
            far: 10000.0,
        },
    };

    'mainloop: loop {
        for event in events.poll_iter() {
            match event {
                Event::Quit { .. } => {
                    break 'mainloop;
                }
                _ => (),
            }
        }

        for node in scene.active_mut() {
            node.local.pos.0[1] += unsafe { sdl2::libc::rand() % 4 } as f32;
            if node.local.pos.0[1] > 1080.0 {
                node.local.pos.0[1] = -32.0;
            }
        }

        scene.update();

        {
            let mut pass = gfx.pass(PassSettings {
                target: Target::Screen,
                camera: &camera,
            });

            pass.clear_all();
            pass.draw(scene.drawables());
        }

        win.gl_swap_window();

        frames += 1.0;
        let now = Instant::now();
        let delta = now.duration_since(last);
        if delta > Duration::from_secs(5) {
            last = now;
            log::debug!("fps: {}", frames / delta.as_secs_f32());
            frames = 0.0;
        }
    }
}
