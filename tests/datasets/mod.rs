//! The benchmarks too large to keep here, fetched the first time a test
//! asks for one and kept for every run after in the target directory,
//! under `target/tmp/datasets`, where Cargo has integration tests keep
//! what they make; `cargo clean` throws them away with the rest of the
//! build. An archive is checked against the SHA-256 recorded for it
//! before it is unpacked, so that a download cut short, or an archive
//! changed since it was recorded, is refused rather than measured; and it
//! is unpacked under another name and moved into its place whole, so that
//! a run stopped part of the way leaves nothing to be taken for the
//! benchmark. Deleting an archive's directory fetches it again.

use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// An archive published elsewhere, and where it is kept once fetched.
pub struct Archive {
    /// The directory under `target/tmp/datasets` it is unpacked into.
    pub name: &'static str,
    pub url: &'static str,
    /// The archive's SHA-256, in hexadecimal.
    pub sha256: &'static str,
}

/// Where the archives are kept: `datasets` in the target directory's
/// `tmp`, wherever `CARGO_TARGET_DIR` puts the target directory.
pub fn kept() -> PathBuf {
    Path::new(env!("CARGO_TARGET_TMPDIR")).join("datasets")
}

/// The directory `archive` unpacks to, fetched and unpacked first when
/// it is not already kept.
pub fn fetched(archive: &Archive) -> PathBuf {
    let datasets = kept();
    let dir = datasets.join(archive.name);
    if dir.is_dir() {
        return dir;
    }
    // What a run stopped part of the way left behind is thrown away.
    let zip = datasets.join(format!("{}.zip.part", archive.name));
    let unpacking = datasets.join(format!("{}.part", archive.name));
    let _ = fs::remove_file(&zip);
    let _ = fs::remove_dir_all(&unpacking);
    fs::create_dir_all(&datasets).unwrap_or_else(|e| panic!("{}: {e}", datasets.display()));

    eprintln!("fetching {} into {}", archive.url, dir.display());
    let response = ureq::get(archive.url).call().unwrap_or_else(|e| panic!("{}: {e}", archive.url));
    io::copy(&mut response.into_body().into_reader(), &mut File::create(&zip).unwrap_or_else(|e| panic!("{}: {e}", zip.display())))
        .unwrap_or_else(|e| panic!("{}: {e}", archive.url));
    let mut hasher = Sha256::new();
    io::copy(&mut File::open(&zip).unwrap_or_else(|e| panic!("{}: {e}", zip.display())), &mut hasher).unwrap_or_else(|e| panic!("{}: {e}", zip.display()));
    let sha256 = format!("{:x}", hasher.finalize());
    if sha256 != archive.sha256 {
        let _ = fs::remove_file(&zip);
        panic!("{}: the archive's SHA-256 is {sha256}, where {} was recorded for it", archive.url, archive.sha256);
    }
    zip::ZipArchive::new(File::open(&zip).unwrap_or_else(|e| panic!("{}: {e}", zip.display())))
        .and_then(|mut archived| archived.extract(&unpacking))
        .unwrap_or_else(|e| panic!("{}: {e}", zip.display()));
    fs::rename(&unpacking, &dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    let _ = fs::remove_file(&zip);
    dir
}
