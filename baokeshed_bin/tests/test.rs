use duct::cmd;
use std::fs;
use std::path::PathBuf;
use tempfile::tempdir;

pub fn baokeshed_exe() -> PathBuf {
    assert_cmd::cargo::cargo_bin("baokeshed")
}

#[test]
fn test_hash_one() {
    let expected = baokeshed::hash(b"foo").to_hex();
    let output = cmd!(baokeshed_exe()).stdin_bytes("foo").read().unwrap();
    assert_eq!(&*expected, &*output);
}

#[test]
fn test_hash_many() {
    let dir = tempdir().unwrap();
    let file1 = dir.path().join("file1");
    fs::write(&file1, b"foo").unwrap();
    let file2 = dir.path().join("file2");
    fs::write(&file2, b"bar").unwrap();
    let output = cmd!(baokeshed_exe(), &file1, &file2, "-")
        .stdin_bytes("baz")
        .read()
        .unwrap();
    let foo_hash = baokeshed::hash(b"foo");
    let bar_hash = baokeshed::hash(b"bar");
    let baz_hash = baokeshed::hash(b"baz");
    let expected = format!(
        "{}  {}\n{}  {}\n{}  -",
        foo_hash.to_hex(),
        file1.to_string_lossy(),
        bar_hash.to_hex(),
        file2.to_string_lossy(),
        baz_hash.to_hex(),
    );
    assert_eq!(expected, output);
}

#[test]
fn test_hash_length() {
    let mut xof = baokeshed::Hasher::new().update(b"foo").finalize_xof();
    let mut expected = String::new();
    expected.push_str(&hex::encode(&xof.read()[..]));
    expected.push_str(&hex::encode(&xof.read()[..36]));
    let output = cmd!(baokeshed_exe(), "--length=100")
        .stdin_bytes("foo")
        .read()
        .unwrap();
    assert_eq!(&*expected, &*output);
}

#[test]
fn test_hash_key() {
    let key = [42; baokeshed::KEY_LEN];
    let expected = baokeshed::keyed_hash(b"foo", &key).to_hex();
    let output = cmd!(baokeshed_exe(), "--key", hex::encode(&key))
        .stdin_bytes("foo")
        .read()
        .unwrap();
    assert_eq!(&*expected, &*output);
}

#[test]
fn test_derive_key() {
    let key = &[99; baokeshed::KEY_LEN];
    let expected = baokeshed::derive_key(key, b"context").to_hash().to_hex();
    let output = cmd!(baokeshed_exe(), "--derive-key", hex::encode(key))
        .stdin_bytes("context")
        .read()
        .unwrap();
    assert_eq!(&*expected, &*output);
}
