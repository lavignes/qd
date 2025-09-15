use std::{
    fmt::Write,
    process,
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, Builder, JoinHandle},
    time::{Duration, Instant},
};

use crossbeam_channel::{Receiver, Sender};
use log::{Level, LevelFilter, Log, Metadata, Record};

const QUEUE_SIZE: usize = 32;
const DEFAULT_BUFFER_SIZE: usize = 256;
static RUNNING: AtomicBool = AtomicBool::new(true);

#[inline]
fn level(lvl: Level) -> &'static str {
    match lvl {
        Level::Error => "\x1B[35mERROR\x1B[0m",
        Level::Warn => "\x1B[33m WARN\x1B[0m",
        Level::Info => "\x1B[34m INFO\x1B[0m",
        Level::Debug => "\x1B[32mDEBUG\x1B[0m",
        Level::Trace => "\x1B[36mTRACE\x1B[0m",
    }
}

struct AsyncLogger {
    tx: Sender<String>,
    brx: Receiver<String>,
    hnd: Mutex<Option<JoinHandle<()>>>,
    start: Instant,
}

impl Log for AsyncLogger {
    #[inline]
    fn enabled(&self, _: &Metadata) -> bool {
        RUNNING.load(Ordering::Relaxed)
    }

    fn log(&self, record: &Record) {
        // fatal is a special case, we leverage custom targets
        if record.metadata().target() == "FATAL" {
            // join with the log thread and handle printing the error here
            if let Ok(mut hnd) = self.hnd.lock() {
                if let Some(hnd) = hnd.take() {
                    RUNNING.store(false, Ordering::Relaxed);
                    hnd.join().unwrap();
                    let current = thread::current();
                    let name = current.name().unwrap_or("???");
                    let time = self.start.elapsed().as_secs_f32();
                    if let (Some(file), Some(line)) = (record.module_path(), record.line()) {
                        eprintln!(
                            "[\x1B[31mFATAL\x1B[0m] {time:09.3} <{name}> {file}:{line}: {}",
                            record.args()
                        )
                    } else {
                        eprintln!(
                            "[\x1B[31mFATAL\x1B[0m] {time:09.3} <{name}> {}",
                            record.args()
                        );
                    }
                    process::exit(1);
                }
            }
        }
        if !self.enabled(record.metadata()) {
            return;
        }
        // TODO: timeout? handle disconnect error?
        let mut buf = self.brx.recv().unwrap();
        let level = level(record.level());
        let current = thread::current();
        let name = current.name().unwrap_or("???");
        let time = self.start.elapsed().as_secs_f32();
        if let (Some(file), Some(line)) = (record.module_path(), record.line()) {
            write!(
                &mut buf,
                "[{level}] {time:09.3} <{name}> {file}:{line}: {}",
                record.args()
            )
            .unwrap();
        } else {
            write!(&mut buf, "[{level}] {time:09.3} <{name}> {}", record.args()).unwrap();
        }
        self.tx.send(buf).unwrap();
    }

    fn flush(&self) {}
}

pub fn init() {
    // queue for ready-to-print buffers
    let (tx, rx): (Sender<String>, _) = crossbeam_channel::bounded(QUEUE_SIZE);
    // queue for ready-to-fill buffers
    let (btx, brx) = crossbeam_channel::bounded(QUEUE_SIZE);
    // pre-allocate a bunch of buffers
    for _ in 0..QUEUE_SIZE {
        btx.send(String::with_capacity(DEFAULT_BUFFER_SIZE))
            .unwrap();
    }
    let hnd = Builder::new()
        .name("logger".into())
        .spawn(move || {
            loop {
                if let Ok(mut buf) = rx.recv_timeout(Duration::from_millis(500)) {
                    eprintln!("{buf}");
                    buf.clear();
                    btx.send(buf).unwrap();
                }
                if !RUNNING.load(Ordering::Relaxed) {
                    break;
                }
            }
        })
        .unwrap();
    // setup log crate
    log::set_logger(Box::leak(Box::new(AsyncLogger {
        tx,
        brx,
        hnd: Mutex::new(Some(hnd)),
        start: Instant::now(),
    })))
    .unwrap();
    log::set_max_level(LevelFilter::Trace);
    log::trace!("Logger initialized");
}
