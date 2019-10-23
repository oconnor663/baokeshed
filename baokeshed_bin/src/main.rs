use failure::Error;
use serde::Deserialize;
use std::cmp;
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::path::{Path, PathBuf};

const VERSION: &str = env!("CARGO_PKG_VERSION");

const USAGE: &str = "
Usage: baokeshed [<inputs>...] [--length=<bytes>] [--key=<hex> | --derive-key=<hex>]
       baokeshed (--help | --version)
";

#[derive(Debug, Deserialize)]
struct Args {
    arg_inputs: Vec<PathBuf>,
    flag_help: bool,
    flag_key: Option<String>,
    flag_derive_key: Option<String>,
    flag_length: Option<u64>,
    flag_version: bool,
}

fn main() -> Result<(), Error> {
    let args: Args = docopt::Docopt::new(USAGE)
        .and_then(|d| d.deserialize())
        .unwrap_or_else(|e| e.exit());
    if args.flag_help {
        print!("{}", USAGE);
    } else if args.flag_version {
        println!("{}", VERSION);
    } else {
        hash(&args)?;
    }

    Ok(())
}

fn hash_one(
    maybe_path: &Option<PathBuf>,
    key: Option<&[u8; baokeshed::KEY_LEN]>,
    derive_key: bool,
) -> Result<baokeshed::Output, Error> {
    let mut input = open_input(maybe_path)?;
    if let Some(map) = maybe_memmap_input(&input)? {
        let output = if let Some(key) = key {
            if derive_key {
                baokeshed::derive_key(key, &map)
            } else {
                baokeshed::keyed_hash_xof(&map, key)
            }
        } else {
            baokeshed::hash_xof(&map)
        };
        Ok(output)
    } else {
        let mut hasher = if let Some(key) = key {
            if derive_key {
                baokeshed::Hasher::new_derive_key(key)
            } else {
                baokeshed::Hasher::new_keyed(key)
            }
        } else {
            baokeshed::Hasher::new()
        };
        baokeshed::copy::copy_wide(&mut input, &mut hasher)?;
        Ok(hasher.finalize_xof())
    }
}

fn print_hex(output: &mut baokeshed::Output, byte_length: u64) {
    let mut hex_bytes_remaining = 2 * byte_length;
    while hex_bytes_remaining > 0 {
        let hex = baokeshed::Hash::from(output.read()).to_hex();
        let take = cmp::min(hex_bytes_remaining, hex.len() as u64);
        print!("{}", &hex[..take as usize]);
        hex_bytes_remaining -= take;
    }
}

fn hash(args: &Args) -> Result<(), Error> {
    let byte_length = args.flag_length.unwrap_or(baokeshed::OUT_LEN as u64);
    let derive_key = args.flag_derive_key.is_some();
    let key = if let Some(key_str) = args.flag_key.as_ref().or(args.flag_derive_key.as_ref()) {
        let key_bytes = hex::decode(key_str)?;
        if key_bytes.len() != baokeshed::KEY_LEN {
            failure::bail!(
                "keys must be {} bytes, found {}",
                baokeshed::KEY_LEN,
                key_bytes.len()
            );
        }
        let mut key_array = [0; baokeshed::KEY_LEN];
        key_array.copy_from_slice(&key_bytes);
        Some(key_array)
    } else {
        None
    };

    if !args.arg_inputs.is_empty() {
        let mut did_error = false;
        for input in args.arg_inputs.iter() {
            let input_str = input.to_string_lossy();
            // As with b2sum or sha1sum, the multi-arg hash loop prints errors and keeps going.
            // This is more convenient for the user in cases like `bao hash *`, where it's common
            // that some of the inputs will error on read e.g. because they're directories.
            match hash_one(&Some(input.clone()), key.as_ref(), derive_key) {
                Ok(mut output) => {
                    print_hex(&mut output, byte_length);
                    if args.arg_inputs.len() > 1 {
                        println!("  {}", input_str);
                    } else {
                        println!();
                    }
                }
                Err(e) => {
                    did_error = true;
                    println!("bao: {}: {}", input_str, e);
                }
            }
        }
        if did_error {
            std::process::exit(1);
        }
    } else {
        let mut output = hash_one(&None, key.as_ref(), derive_key)?;
        print_hex(&mut output, byte_length);
        println!();
    }

    Ok(())
}

fn open_input(maybe_path: &Option<PathBuf>) -> Result<Input, Error> {
    Ok(
        if let Some(ref path) = path_if_some_and_not_dash(maybe_path) {
            Input::File(File::open(path)?)
        } else {
            Input::Stdin
        },
    )
}

enum Input {
    Stdin,
    File(File),
}

impl Read for Input {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match *self {
            Input::Stdin => io::stdin().read(buf),
            Input::File(ref mut file) => file.read(buf),
        }
    }
}

fn path_if_some_and_not_dash(maybe_path: &Option<PathBuf>) -> Option<&Path> {
    if let Some(ref path) = maybe_path {
        if path == Path::new("-") {
            None
        } else {
            Some(path)
        }
    } else {
        None
    }
}

fn maybe_memmap_input(input: &Input) -> Result<Option<memmap::Mmap>, Error> {
    let in_file = match *input {
        Input::Stdin => return Ok(None),
        Input::File(ref file) => file,
    };
    let metadata = in_file.metadata()?;
    Ok(if !metadata.is_file() {
        // Not a real file.
        None
    } else if metadata.len() > isize::max_value() as u64 {
        // Too long to safely map. https://github.com/danburkert/memmap-rs/issues/69
        None
    } else if metadata.len() == 0 {
        // Mapping an empty file currently fails. https://github.com/danburkert/memmap-rs/issues/72
        None
    } else {
        // Explicitly set the length of the memory map, so that filesystem changes can't race to
        // violate the invariants we just checked.
        let map = unsafe {
            memmap::MmapOptions::new()
                .len(metadata.len() as usize)
                .map(&in_file)?
        };
        Some(map)
    })
}
