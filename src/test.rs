use crate::*;

use core::usize;

#[test]
fn test_offset_words() {
    let offset: u64 = (1 << 32) + 2;
    assert_eq!(offset_low(offset), 2);
    assert_eq!(offset_high(offset), 1);
}

#[test]
fn test_block_flags() {
    let block_len: u8 = 0b1110;
    let root_flag: u8 = Flags::ROOT.bits();
    assert_eq!(root_flag, 0b10000000);
    assert_eq!(
        block_flags(block_len, root_flag),
        0b10000000000000000000000000001110
    );
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
];

pub const TEST_CASES_MAX: usize = 8 * CHUNK_LEN + 1;

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
        let context = 23;

        let recursive_hash = hash_keyed_contextified(input, key, context);

        let incremental_hash_all = Hasher::new_keyed_contextified(key, context)
            .update(input)
            .finalize();
        assert_eq!(recursive_hash, incremental_hash_all);

        let mut hasher_one_at_a_time = Hasher::new_keyed_contextified(key, context);
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
    let mut state = iv(&key_words);
    let block = [0; BLOCK_LEN];
    let internal_flags = Flags::CHUNK_START | Flags::CHUNK_END | Flags::ROOT;
    let context = 23;
    portable::compress(&mut state, &block, 0, 0, internal_flags.bits(), context);
    let expected_hash: Hash = bytes_from_state_words(&state).into();

    assert_eq!(expected_hash, hash_keyed_contextified(&[], &key, context));

    let hasher = Hasher::new_keyed_contextified(&key, context);
    assert_eq!(expected_hash, hasher.finalize());
}

#[test]
fn test_one_byte() {
    let mut key = [42; KEY_LEN];
    paint_test_input(&mut key);
    let key_words = words_from_key_bytes(&key);
    let mut state = iv(&key_words);
    let mut block = [0; BLOCK_LEN];
    block[0] = 9;
    let internal_flags = Flags::CHUNK_START | Flags::CHUNK_END | Flags::ROOT;
    let context = 23;
    portable::compress(&mut state, &block, 1, 0, internal_flags.bits(), context);
    let expected_hash: Hash = bytes_from_state_words(&state).into();

    assert_eq!(expected_hash, hash_keyed_contextified(&[9], &key, context));

    let mut hasher = Hasher::new_keyed_contextified(&key, context);
    hasher.update(&[9]);
    assert_eq!(expected_hash, hasher.finalize());
}

type Construction = fn(input_buf: &[u8], key: &[u8; KEY_LEN], flags: Word) -> Hash;

fn exercise_construction(construction: Construction, input_len: usize) {
    let mut input_buf = [0; 65536];
    paint_test_input(&mut input_buf);
    let input = &input_buf[..input_len];

    // Check the default parameters.
    let expected_default_hash = construction(&input, &[0; KEY_LEN], 0);
    assert_eq!(expected_default_hash, hash(&input));
    assert_eq!(expected_default_hash, hash_keyed(&input, &[0; KEY_LEN]));
    assert_eq!(
        expected_default_hash,
        hash_keyed_contextified(&input, &[0; KEY_LEN], 0)
    );
    assert_eq!(
        expected_default_hash,
        hash_keyed_contextified_xof(&input, &[0; KEY_LEN], 0).read()
    );
    assert_eq!(
        expected_default_hash,
        Hasher::new().update(&input).finalize(),
    );
    assert_eq!(
        expected_default_hash,
        Hasher::new_keyed(&[0; KEY_LEN]).update(&input).finalize(),
    );
    assert_eq!(
        expected_default_hash,
        Hasher::new_keyed_contextified(&[0; KEY_LEN], 0)
            .update(&input)
            .finalize(),
    );
    assert_eq!(
        expected_default_hash,
        Hasher::new().update(&input).finalize_xof().read(),
    );

    // Check non-default parameters.
    let key = array_ref!(input, 7, KEY_LEN);
    let context = 23;
    let expected_nondefault_hash = construction(&input, key, context);
    assert_eq!(
        expected_nondefault_hash,
        hash_keyed_contextified(&input, key, context)
    );
    assert_eq!(
        expected_nondefault_hash,
        hash_keyed_contextified_xof(&input, key, context).read(),
    );
    assert_eq!(
        expected_nondefault_hash,
        Hasher::new_keyed_contextified(key, context)
            .update(&input)
            .finalize(),
    );
    assert_eq!(
        expected_nondefault_hash,
        Hasher::new_keyed_contextified(key, context)
            .update(&input)
            .finalize_xof()
            .read(),
    );
}

fn three_blocks_construction(input_buf: &[u8], key: &[u8; KEY_LEN], context: Word) -> Hash {
    let key_words = words_from_key_bytes(&key);
    let mut state = iv(&key_words);

    let block0 = array_ref!(input_buf, 0, BLOCK_LEN);
    portable::compress(
        &mut state,
        &block0,
        BLOCK_LEN as u8,
        0,
        Flags::CHUNK_START.bits(),
        context,
    );

    let block1 = array_ref!(input_buf, BLOCK_LEN, BLOCK_LEN);
    portable::compress(
        &mut state,
        &block1,
        BLOCK_LEN as u8,
        0, // Subsequent blocks keep using the chunk's starting offset.
        0, // Middle blocks have no internal flags.
        context,
    );

    let mut block2 = [0; BLOCK_LEN];
    block2[0] = input_buf[2 * BLOCK_LEN];
    portable::compress(
        &mut state,
        &block2,
        1,
        0, // Subsequent blocks keep using the chunk's starting offset.
        (Flags::CHUNK_END | Flags::ROOT).bits(),
        context,
    );

    bytes_from_state_words(&state).into()
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
    context: Word,
) -> [u8; OUT_LEN] {
    assert_eq!(chunk.len(), CHUNK_LEN);
    let blocks = CHUNK_LEN / BLOCK_LEN;
    let mut state = iv(&key);
    // First block.
    portable::compress(
        &mut state,
        array_ref!(chunk, 0, BLOCK_LEN),
        BLOCK_LEN as u8,
        offset,
        Flags::CHUNK_START.bits(),
        context,
    );
    // Middle blocks.
    for block_index in 1..blocks - 1 {
        portable::compress(
            &mut state,
            array_ref!(chunk, block_index * BLOCK_LEN, BLOCK_LEN),
            BLOCK_LEN as u8,
            offset,
            0,
            context,
        );
    }
    // Last block.
    portable::compress(
        &mut state,
        array_ref!(chunk, (blocks - 1) * BLOCK_LEN, BLOCK_LEN),
        BLOCK_LEN as u8,
        offset,
        Flags::CHUNK_END.bits(),
        context,
    );
    bytes_from_state_words(&state)
}

fn three_chunks_construction(input_buf: &[u8], key: &[u8; KEY_LEN], context: Word) -> Hash {
    let key_words = words_from_key_bytes(&key);

    // The first chunk.
    let chunk0_out = hash_whole_chunk_for_testing(&input_buf[..CHUNK_LEN], &key_words, 0, context);

    // The second chunk.
    let chunk1_out = hash_whole_chunk_for_testing(
        &input_buf[CHUNK_LEN..][..CHUNK_LEN],
        &key_words,
        CHUNK_LEN as u64,
        context,
    );

    // The third and final chunk is one byte.
    let mut chunk2_block = [0; BLOCK_LEN];
    chunk2_block[0] = input_buf[2 * CHUNK_LEN];
    let mut chunk2_state = iv(&key_words);
    portable::compress(
        &mut chunk2_state,
        &chunk2_block,
        1,
        2 * CHUNK_LEN as u64,
        (Flags::CHUNK_START | Flags::CHUNK_END).bits(),
        context,
    );
    let chunk2_out = bytes_from_state_words(&chunk2_state);

    // The parent of the first two chunks.
    let mut left_parent_state = iv(&key_words);
    let mut left_parent_block = [0; BLOCK_LEN];
    left_parent_block[..OUT_LEN].copy_from_slice(&chunk0_out);
    left_parent_block[OUT_LEN..].copy_from_slice(&chunk1_out);
    portable::compress(
        &mut left_parent_state,
        &left_parent_block,
        BLOCK_LEN as u8,
        0,
        Flags::PARENT.bits(),
        context,
    );
    let left_parent_out = bytes_from_state_words(&left_parent_state);

    // The root node.
    let mut root_state = iv(&key_words);
    let mut root_block = [0; BLOCK_LEN];
    root_block[..OUT_LEN].copy_from_slice(&left_parent_out);
    root_block[OUT_LEN..].copy_from_slice(&chunk2_out);
    portable::compress(
        &mut root_state,
        &root_block,
        BLOCK_LEN as u8,
        0,
        (Flags::PARENT | Flags::ROOT).bits(),
        context,
    );
    bytes_from_state_words(&root_state).into()
}

#[test]
fn test_three_chunks() {
    exercise_construction(three_chunks_construction, 2 * CHUNK_LEN + 1);
}

#[test]
fn test_default_key() {
    let default_key = &[0; KEY_LEN];

    let expected_hash = hash_keyed(b"abc", &default_key);

    assert_eq!(expected_hash, hash(b"abc"));

    let mut hasher = Hasher::new();
    hasher.update(b"abc");
    assert_eq!(expected_hash, hasher.finalize());
}

#[test]
fn test_xof_output() {
    let input = b"abc";
    let key = &[42; KEY_LEN];
    let context = 23;
    let expected_hash = hash_keyed_contextified(input, key, context);

    let mut xof = hash_keyed_contextified_xof(input, key, context);
    let mut hasher_xof = Hasher::new_keyed_contextified(key, context)
        .update(input)
        .finalize_xof();

    let first_bytes = xof.read();
    assert_eq!(first_bytes, hasher_xof.read());
    assert_eq!(&first_bytes, expected_hash.as_bytes());

    let second_bytes = xof.read();
    assert_eq!(second_bytes, hasher_xof.read());
    assert!(first_bytes != second_bytes);

    let third_bytes = xof.read();
    assert_eq!(third_bytes, hasher_xof.read());
    assert!(first_bytes != third_bytes);
    assert!(second_bytes != third_bytes);
}
