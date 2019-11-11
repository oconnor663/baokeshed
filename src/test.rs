use crate::*;

use core::usize;

// Interesting input lengths to run tests on.
pub const TEST_CASES: &[usize] = &[
    0,
    1,
    CHUNK_LEN - 1,
    CHUNK_LEN,
    CHUNK_LEN + 1,
    2 * CHUNK_LEN,
    2 * CHUNK_LEN + 1,
    3 * CHUNK_LEN,
    3 * CHUNK_LEN + 1,
    4 * CHUNK_LEN,
    4 * CHUNK_LEN + 1,
    5 * CHUNK_LEN,
    5 * CHUNK_LEN + 1,
    6 * CHUNK_LEN,
    6 * CHUNK_LEN + 1,
    7 * CHUNK_LEN,
    7 * CHUNK_LEN + 1,
    8 * CHUNK_LEN,
    8 * CHUNK_LEN + 1,
    16 * CHUNK_LEN, // AVX512's bandwidth
    31 * CHUNK_LEN, // 16 + 8 + 4 + 2 + 1
];

pub const TEST_CASES_MAX: usize = 31 * CHUNK_LEN;

// Paint a byte pattern that won't repeat, so that we don't accidentally
// miss buffer offset bugs.
pub fn paint_test_input(buf: &mut [u8]) {
    let mut offset = 0;
    let mut counter: u32 = 1;
    while offset < buf.len() {
        let bytes = counter.to_le_bytes();
        let take = cmp::min(bytes.len(), buf.len() - offset);
        buf[offset..][..take].copy_from_slice(&bytes[..take]);
        counter += 1;
        offset += take;
    }
}

#[test]
fn test_offset_words() {
    let offset: u64 = (1 << 32) + 2;
    assert_eq!(offset_low(offset), 2);
    assert_eq!(offset_high(offset), 1);
}

#[test]
fn test_largest_power_of_two_leq() {
    let input_output = &[
        // The zero case is nonsensical, but it does work.
        (0, 1),
        (1, 1),
        (2, 2),
        (3, 2),
        (4, 4),
        (5, 4),
        (6, 4),
        (7, 4),
        (8, 8),
        // the largest possible usize
        (usize::MAX, (usize::MAX >> 1) + 1),
    ];
    for &(input, output) in input_output {
        assert_eq!(
            output,
            largest_power_of_two_leq(input),
            "wrong output for n={}",
            input
        );
    }
}

#[test]
fn test_left_len() {
    let input_output = &[
        (CHUNK_LEN + 1, CHUNK_LEN),
        (2 * CHUNK_LEN - 1, CHUNK_LEN),
        (2 * CHUNK_LEN, CHUNK_LEN),
        (2 * CHUNK_LEN + 1, 2 * CHUNK_LEN),
        (4 * CHUNK_LEN - 1, 2 * CHUNK_LEN),
        (4 * CHUNK_LEN, 2 * CHUNK_LEN),
        (4 * CHUNK_LEN + 1, 4 * CHUNK_LEN),
    ];
    for &(input, output) in input_output {
        assert_eq!(left_len(input), output);
    }
}

#[test]
fn test_recursive_incremental_same() {
    // This is a pretty large stack array. I don't want to use a Vec here,
    // because these tests need to be no_std compatible. If this becomes a
    // problem, we can make this one an integration test instead, which would
    // let it use std:: even when the crate doesn't.
    let mut input_buf = [0; TEST_CASES_MAX];
    paint_test_input(&mut input_buf);
    for &case in TEST_CASES {
        let input = &input_buf[..case];
        let key = array_ref!(input_buf, 0, KEY_LEN);

        let recursive_hash = keyed_hash(input, key);

        let incremental_hash_all = Hasher::new_keyed(key).update(input).finalize();
        assert_eq!(recursive_hash, incremental_hash_all);

        let mut hasher_one_at_a_time = Hasher::new_keyed(key);
        for &byte in input {
            hasher_one_at_a_time.update(&[byte]);
        }
        assert_eq!(recursive_hash, hasher_one_at_a_time.finalize());
    }
}

#[test]
fn test_zero_bytes() {
    let mut key = [42; KEY_LEN];
    paint_test_input(&mut key);
    let key_words = words_from_key_bytes(&key);
    let mut cv = key_words;
    let block = [0; BLOCK_LEN];
    let flags = Flags::CHUNK_START | Flags::CHUNK_END | Flags::ROOT | Flags::KEYED_HASH;
    portable::compress(&mut cv, &block, 0, 0, flags.bits());
    let expected_hash: Hash = bytes_from_state_words(&cv).into();

    assert_eq!(expected_hash, keyed_hash(&[], &key,));

    let hasher = Hasher::new_keyed(&key);
    assert_eq!(expected_hash, hasher.finalize());
}

#[test]
fn test_one_byte() {
    let mut key = [42; KEY_LEN];
    paint_test_input(&mut key);
    let key_words = words_from_key_bytes(&key);
    let mut cv = key_words;
    let mut block = [0; BLOCK_LEN];
    block[0] = 9;
    let flags = Flags::CHUNK_START | Flags::CHUNK_END | Flags::ROOT | Flags::KEYED_HASH;
    portable::compress(&mut cv, &block, 1, 0, flags.bits());
    let expected_hash: Hash = bytes_from_state_words(&cv).into();

    assert_eq!(expected_hash, keyed_hash(&[9], &key,));

    let mut hasher = Hasher::new_keyed(&key);
    hasher.update(&[9]);
    assert_eq!(expected_hash, hasher.finalize());
}

type Construction = fn(input_buf: &[u8], key: &[u8; KEY_LEN], flags: Flags) -> Hash;

fn exercise_construction(construction: Construction, input_len: usize) {
    let mut input_buf = [0; 65536];
    paint_test_input(&mut input_buf);
    let input = &input_buf[..input_len];

    // Check the default hash.
    let expected_default_hash = construction(&input, &bytes_from_state_words(&IV), Flags::empty());
    assert_eq!(expected_default_hash, hash(&input));
    assert_eq!(
        expected_default_hash,
        Hasher::new().update(&input).finalize(),
    );
    assert_eq!(
        expected_default_hash,
        Hasher::new().update(&input).finalize_xof().to_hash(),
    );

    // Check the keyed hash.
    let key = array_ref!(input, 7, KEY_LEN);
    let expected_keyed_hash = construction(&input, key, Flags::KEYED_HASH);
    assert_eq!(expected_keyed_hash, keyed_hash(&input, key));
    assert_eq!(
        expected_keyed_hash,
        Hasher::new_keyed(key).update(&input).finalize(),
    );
    assert_eq!(
        expected_keyed_hash,
        Hasher::new_keyed(key)
            .update(&input)
            .finalize_xof()
            .to_hash(),
    );

    // Check the KDF.
    let key = array_ref!(input, 7, KEY_LEN);
    let expected_kdf_out = construction(&input, key, Flags::DERIVE_KEY);
    assert_eq!(expected_kdf_out, derive_key_xof(key, &input).to_hash());
    assert_eq!(
        expected_kdf_out,
        Hasher::new_derive_key(key).update(&input).finalize(),
    );
    assert_eq!(
        expected_kdf_out,
        Hasher::new_derive_key(key)
            .update(&input)
            .finalize_xof()
            .to_hash(),
    );
}

fn three_blocks_construction(input_buf: &[u8], key: &[u8; KEY_LEN], flags: Flags) -> Hash {
    let key_words = words_from_key_bytes(&key);
    let mut cv = key_words;

    let block0 = array_ref!(input_buf, 0, BLOCK_LEN);
    portable::compress(
        &mut cv,
        &block0,
        BLOCK_LEN as u8,
        0,
        (flags | Flags::CHUNK_START).bits(),
    );

    let block1 = array_ref!(input_buf, BLOCK_LEN, BLOCK_LEN);
    portable::compress(
        &mut cv,
        &block1,
        BLOCK_LEN as u8,
        0, // Subsequent blocks keep using the chunk's starting offset.
        flags.bits(),
    );

    let mut block2 = [0; BLOCK_LEN];
    block2[0] = input_buf[2 * BLOCK_LEN];
    portable::compress(
        &mut cv,
        &block2,
        1,
        0, // Subsequent blocks keep using the chunk's starting offset.
        (flags | Flags::CHUNK_END | Flags::ROOT).bits(),
    );

    bytes_from_state_words(&cv).into()
}

#[test]
fn test_three_blocks() {
    exercise_construction(three_blocks_construction, 2 * BLOCK_LEN + 1);
}

// This is equivalent to lib.rs::hash_one_chunk. Calling that function from
// tests might hide bugs, so we reimplement a simple version of it here.
fn hash_whole_chunk_for_testing(
    chunk: &[u8],
    key: &[Word; 8],
    offset: u64,
    flags: Flags,
) -> [u8; OUT_LEN] {
    assert_eq!(chunk.len(), CHUNK_LEN);
    let blocks = CHUNK_LEN / BLOCK_LEN;
    let mut cv = *key;
    // First block.
    portable::compress(
        &mut cv,
        array_ref!(chunk, 0, BLOCK_LEN),
        BLOCK_LEN as u8,
        offset,
        (flags | Flags::CHUNK_START).bits(),
    );
    // Middle blocks.
    for block_index in 1..blocks - 1 {
        portable::compress(
            &mut cv,
            array_ref!(chunk, block_index * BLOCK_LEN, BLOCK_LEN),
            BLOCK_LEN as u8,
            offset,
            flags.bits(),
        );
    }
    // Last block.
    portable::compress(
        &mut cv,
        array_ref!(chunk, (blocks - 1) * BLOCK_LEN, BLOCK_LEN),
        BLOCK_LEN as u8,
        offset,
        (flags | Flags::CHUNK_END).bits(),
    );
    bytes_from_state_words(&cv)
}

fn three_chunks_construction(input_buf: &[u8], key: &[u8; KEY_LEN], flags: Flags) -> Hash {
    let key_words = words_from_key_bytes(&key);

    // The first chunk.
    let chunk0_out = hash_whole_chunk_for_testing(&input_buf[..CHUNK_LEN], &key_words, 0, flags);

    // The second chunk.
    let chunk1_out = hash_whole_chunk_for_testing(
        &input_buf[CHUNK_LEN..][..CHUNK_LEN],
        &key_words,
        CHUNK_LEN as u64,
        flags,
    );

    // The third and final chunk is one byte.
    let mut chunk2_block = [0; BLOCK_LEN];
    chunk2_block[0] = input_buf[2 * CHUNK_LEN];
    let mut chunk2_cv = key_words;
    portable::compress(
        &mut chunk2_cv,
        &chunk2_block,
        1,
        2 * CHUNK_LEN as u64,
        (flags | Flags::CHUNK_START | Flags::CHUNK_END).bits(),
    );
    let chunk2_out = bytes_from_state_words(&chunk2_cv);

    // The parent of the first two chunks.
    let mut left_parent_cv = key_words;
    let mut left_parent_block = [0; BLOCK_LEN];
    left_parent_block[..OUT_LEN].copy_from_slice(&chunk0_out);
    left_parent_block[OUT_LEN..].copy_from_slice(&chunk1_out);
    portable::compress(
        &mut left_parent_cv,
        &left_parent_block,
        BLOCK_LEN as u8,
        0,
        (flags | Flags::PARENT).bits(),
    );
    let left_parent_out = bytes_from_state_words(&left_parent_cv);

    // The root node.
    let mut root_cv = key_words;
    let mut root_block = [0; BLOCK_LEN];
    root_block[..OUT_LEN].copy_from_slice(&left_parent_out);
    root_block[OUT_LEN..].copy_from_slice(&chunk2_out);
    portable::compress(
        &mut root_cv,
        &root_block,
        BLOCK_LEN as u8,
        0,
        (flags | Flags::PARENT | Flags::ROOT).bits(),
    );
    bytes_from_state_words(&root_cv).into()
}

#[test]
fn test_three_chunks() {
    exercise_construction(three_chunks_construction, 2 * CHUNK_LEN + 1);
}

#[test]
fn test_xof_output() {
    let input = b"abc";
    let key = &[42; KEY_LEN];
    let expected_hash = keyed_hash(input, key);

    let mut xof = Hasher::new_keyed(key).update(input).finalize_xof();

    let first_bytes = xof.read();
    assert_eq!(&first_bytes[..OUT_LEN], expected_hash.as_bytes());
    assert!(&first_bytes[OUT_LEN..] != expected_hash.as_bytes());

    let second_bytes = xof.read();
    assert!(&first_bytes[..] != &second_bytes[..]);

    let third_bytes = xof.read();
    assert!(&first_bytes[..] != &third_bytes[..]);
    assert!(&second_bytes[..] != &third_bytes[..]);
}

#[test]
fn test_domain_separation() {
    let h1 = hash(b"foo");
    let h2 = keyed_hash(b"foo", &bytes_from_state_words(&IV));
    let h3 = derive_key_xof(&bytes_from_state_words(&IV), b"foo").to_hash();
    assert!(h1 != h2);
    assert!(h2 != h3);
}

#[test]
fn test_regular_xof_match() {
    assert_eq!(hash(b"foo"), hash_xof(b"foo").to_hash());
    assert_eq!(
        keyed_hash(b"foo", &[5; KEY_LEN]),
        keyed_hash_xof(b"foo", &[5; KEY_LEN]).to_hash()
    );
    assert_eq!(
        &derive_key(&[5; KEY_LEN], b"foo")[..],
        &derive_key_xof(&[5; KEY_LEN], b"foo").read()[..OUT_LEN],
    );
}

#[test]
fn test_lib_against_reference_impl() {
    let mut key = [0; KEY_LEN];
    paint_test_input(&mut key);

    for &case in TEST_CASES {
        dbg!(case);
        let mut input = vec![0; case];
        paint_test_input(&mut input);

        // all at once
        let mut hasher = reference_impl::Hasher::new();
        hasher.update(&input);
        let output = hasher.finalize();
        assert_eq!(hash(&input), output);

        // one byte at a time
        let mut hasher = reference_impl::Hasher::new();
        for &byte in &input {
            hasher.update(&[byte]);
        }
        let output = hasher.finalize();
        assert_eq!(hash(&input), output);

        // keyed
        let mut hasher = reference_impl::Hasher::new_keyed(&key);
        hasher.update(&input);
        let output = hasher.finalize();
        assert_eq!(keyed_hash(&input, &key), output);

        // derive key
        let mut hasher = reference_impl::Hasher::new_derive_key(&key);
        hasher.update(&input);
        let output = hasher.finalize();
        assert_eq!(derive_key(&key, &input), output);

        // extended output
        let mut hasher = reference_impl::Hasher::new_keyed(&key);
        hasher.update(&input);
        let mut output = [0; 300];
        hasher.finalize_extended(&mut output);
        let mut expected_output = Vec::new();
        let mut xof = keyed_hash_xof(&input, &key);
        while expected_output.len() < output.len() {
            expected_output.extend_from_slice(&xof.read());
        }
        assert_eq!(&expected_output[..output.len()], &output[..]);
    }
}
