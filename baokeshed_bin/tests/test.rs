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
    let output = cmd!(baokeshed_exe()).input("foo").read().unwrap();
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
        .input("baz")
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
    let mut xof = baokeshed::hash_xof(b"foo");
    let mut expected = String::new();
    expected.push_str(baokeshed::Hash::from(xof.read()).to_hex().as_ref());
    expected.push_str(baokeshed::Hash::from(xof.read()).to_hex().as_ref());
    let output = cmd!(baokeshed_exe(), "--length=64")
        .input("foo")
        .read()
        .unwrap();
    assert_eq!(&*expected, &*output);
}
