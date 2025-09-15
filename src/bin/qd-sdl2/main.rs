use std::time::{Duration, Instant};

use qd::{
    gfx::{Camera, Gfx, Node, Proj, Target, Vtx},
    math::{UV2, V3, V4, Xform3},
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

    let mut gfx = Gfx::new(&qd::gfx::Settings {
        size: UV2([1920, 1080]),
    });

    let mut camera = Camera {
        pos: V3([-10.0, 0.0, 1.0]),
        at: V3([-10.0, 0.0, 0.0]),
        proj: Proj::Ortho {
            size: UV2([1920, 1080]),
            near: 0.0,
            far: 10000.0,
        },
    };

    let mesh = gfx.mesh_alloc(4, 6);
    {
        let (mut vmap, mut imap) = gfx.mesh_map(mesh);
        vmap.write(&[
            Vtx {
                pos: V3([-10.0, -10.0, 0.0]),
                color: V4::splat(1.0),
                ..Default::default()
            },
            Vtx {
                pos: V3([-10.0, 10.0, 0.0]),
                color: V4::splat(1.0),
                ..Default::default()
            },
            Vtx {
                pos: V3([10.0, -10.0, 0.0]),
                color: V4::splat(1.0),
                ..Default::default()
            },
            Vtx {
                pos: V3([10.0, 10.0, 0.0]),
                color: V4::splat(1.0),
                ..Default::default()
            },
        ]);
        imap.write(&[0, 1, 2, 2, 1, 3]);
    }

    let tex = gfx.tex_alloc();
    {
        let mut tmap = gfx.tex_map(tex);
        tmap.write(&vec![0xFFFFFFFF; 256 * 266]);
    }

    let mut events = qd::ensure!(sdl.event_pump());

    let mut frames = 0.0;
    let mut last = Instant::now();

    'mainloop: loop {
        for event in events.poll_iter() {
            match event {
                Event::Quit { .. } => {
                    break 'mainloop;
                }
                _ => (),
            }
        }

        {
            let mut pass = gfx.pass(Target::Screen, &camera);
            pass.clear();
            pass.draw(&[Node {
                mesh,
                tex,
                blend: V4::splat(1.0),
                xform: Xform3::IDENTITY,
            }]);
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
