use failure::Error;
use serde::Deserialize;
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::path::{Path, PathBuf};

const VERSION: &str = env!("CARGO_PKG_VERSION");

const USAGE: &str = "
Usage: baokeshed [<inputs>...]
       baokeshed (--help | --version)
";

#[derive(Debug, Deserialize)]
struct Args {
    arg_inputs: Vec<PathBuf>,
    flag_help: bool,
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

fn hash_one(maybe_path: &Option<PathBuf>) -> Result<baokeshed::Hash, Error> {
    let mut input = open_input(maybe_path)?;
    if let Some(map) = maybe_memmap_input(&input)? {
        Ok(baokeshed::hash(&map))
    } else {
        let mut hasher = baokeshed::Hasher::new();
        baokeshed::copy::copy_wide(&mut input, &mut hasher)?;
        Ok(hasher.finalize())
    }
}

fn hash(args: &Args) -> Result<(), Error> {
    if !args.arg_inputs.is_empty() {
        let mut did_error = false;
        for input in args.arg_inputs.iter() {
            let input_str = input.to_string_lossy();
            // As with b2sum or sha1sum, the multi-arg hash loop prints errors and keeps going.
            // This is more convenient for the user in cases like `bao hash *`, where it's common
            // that some of the inputs will error on read e.g. because they're directories.
            match hash_one(&Some(input.clone())) {
                Ok(hash) => {
                    if args.arg_inputs.len() > 1 {
                        println!("{}  {}", hash.to_hex(), input_str);
                    } else {
                        println!("{}", hash.to_hex());
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
        let hash = hash_one(&None)?;
        println!("{}", hash.to_hex());
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
