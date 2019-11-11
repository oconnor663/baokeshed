//! Utilities for copying from a reader more efficently than with
//! [`std::io::copy`](https://doc.rust-lang.org/std/io/fn.copy.html).

use crate::{CHUNK_LEN, MAX_SIMD_DEGREE};
use std::io;

/// An efficient buffer size for [`Hasher`](../struct.Hasher.html).
///
/// The streaming implementation is single threaded, but it uses SIMD
/// parallelism to get good performance. To avoid unnecessary copying, it
/// relies on the caller to use a buffer size large enough to occupy all the
/// SIMD lanes on the machine. This constant, or an integer multiple of it, is
/// an optimal size.
///
/// On x86 for example, the AVX2 instruction set supports hashing 8 chunks in
/// parallel. Chunks are 2048 bytes each, so `WIDE_BUF_LEN` is currently
/// 16384 bytes. When Rust adds support for AVX512, the value of `WIDE_BUF_LEN`
/// on x86 will double to 32768 bytes. It's not expected to grow any larger
/// than that for the foreseeable future, so on not-very-space-constrained
/// platforms it's possible to use `WIDE_BUF_LEN` as the size of a stack array.
/// If this constant grows larger than 65536 on any platform, it will be
/// considered a backwards-incompatible change, and it will be accompanied by a
/// major version bump.
pub const WIDE_BUF_LEN: usize = MAX_SIMD_DEGREE * CHUNK_LEN;

// This is an implementation detail of libstd, and if it changes there we
// should update it here. This is covered in the tests.
#[allow(dead_code)]
const STD_DEFAULT_BUF_LEN: usize = 8192;

// Const functions can't use if-statements yet, which means that cmp::min and
// cmp::max aren't const. So we have to hardcode the buffer size that copy is
// going to use. This is covered in the tests, and we can replace this with
// cmp::max in the future when it's const.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const COPY_BUF_LEN: usize = WIDE_BUF_LEN;
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
const COPY_BUF_LEN: usize = STD_DEFAULT_BUF_LEN;

/// Copies the entire contents of a reader into a writer, just like
/// [`std::io::copy`](https://doc.rust-lang.org/std/io/fn.copy.html), using a
/// buffer size that's more efficient for [`Hasher`](../struct.Hasher.html).
///
/// The standard `copy` function uses a buffer size that's too small to get
/// good SIMD performance on x86. This function uses a buffer size that's a
/// multiple of [`WIDE_BUF_LEN`](constant.WIDE_BUF_LEN.html).
pub fn copy_wide<R: ?Sized, W: ?Sized>(reader: &mut R, writer: &mut W) -> io::Result<u64>
where
    R: io::Read,
    W: io::Write,
{
    let mut buffer = [0; COPY_BUF_LEN];
    let mut total = 0;
    loop {
        match reader.read(&mut buffer) {
            Ok(0) => return Ok(total),
            Ok(n) => {
                writer.write_all(&buffer[..n])?;
                total += n as u64;
            }
            Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_copy_wide() {
        let mut src = vec![0; 1_000_000];
        crate::test_shared::paint_test_input(&mut src);
        let mut dest = Vec::with_capacity(src.len());
        copy_wide(&mut &src[..], &mut dest).unwrap();
        assert_eq!(src, dest);
    }

    #[test]
    fn test_copy_buffer_sizes() {
        // Check that STD_DEFAULT_BUF_LEN is actually what libstd is using.
        use io::BufRead;
        let bytes = [0; 2 * STD_DEFAULT_BUF_LEN];
        let mut buffered_reader = io::BufReader::new(&bytes[..]);
        let internal_buf = buffered_reader.fill_buf().unwrap();
        assert_eq!(internal_buf.len(), STD_DEFAULT_BUF_LEN);
        assert!(internal_buf.len() < bytes.len());

        // Check that COPY_BUF_LEN is at least STD_DEFAULT_BUF_LEN.
        assert!(COPY_BUF_LEN >= STD_DEFAULT_BUF_LEN);

        // Check that COPY_BUF_LEN is a multiple of WIDE_BUF_LEN.
        assert!(COPY_BUF_LEN >= WIDE_BUF_LEN);
        assert_eq!(0, COPY_BUF_LEN % WIDE_BUF_LEN);
    }
}
