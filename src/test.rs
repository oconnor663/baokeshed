use crate::*;

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
        // the largest possible u64
        (0xffffffffffffffff, 0x8000000000000000),
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
        (CHUNK_BYTES + 1, CHUNK_BYTES),
        (2 * CHUNK_BYTES - 1, CHUNK_BYTES),
        (2 * CHUNK_BYTES, CHUNK_BYTES),
        (2 * CHUNK_BYTES + 1, 2 * CHUNK_BYTES),
        (4 * CHUNK_BYTES - 1, 2 * CHUNK_BYTES),
        (4 * CHUNK_BYTES, 2 * CHUNK_BYTES),
        (4 * CHUNK_BYTES + 1, 4 * CHUNK_BYTES),
    ];
    for &(input, output) in input_output {
        assert_eq!(left_len(input), output);
    }
}

// Interesting input lengths to run tests on.
pub const TEST_CASES: &[usize] = &[
    0,
    1,
    10,
    CHUNK_BYTES - 1,
    CHUNK_BYTES,
    CHUNK_BYTES + 1,
    2 * CHUNK_BYTES - 1,
    2 * CHUNK_BYTES,
    2 * CHUNK_BYTES + 1,
    3 * CHUNK_BYTES - 1,
    3 * CHUNK_BYTES,
    3 * CHUNK_BYTES + 1,
    4 * CHUNK_BYTES - 1,
    4 * CHUNK_BYTES,
    4 * CHUNK_BYTES + 1,
    8 * CHUNK_BYTES - 1,
    8 * CHUNK_BYTES,
    8 * CHUNK_BYTES + 1,
];

// Paint a byte pattern that won't repeat, so that we don't accidentally
// miss buffer offset bugs.
fn paint_test_input(buf: &mut [u8]) {
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
    let mut input_buf = [0; 65536];
    paint_test_input(&mut input_buf);
    for &case in TEST_CASES {
        let input = &input_buf[..case];
        let key = array_ref!(input_buf, 0, KEY_BYTES);

        let recursive_hash = hash_keyed(input, key);

        let mut hasher_all = Hasher::new_keyed(key);
        hasher_all.append(input);
        let incremental_hash_all = hasher_all.finalize();
        assert_eq!(recursive_hash, incremental_hash_all);

        let mut hasher_one_at_a_time = Hasher::new_keyed(key);
        for &byte in input {
            hasher_one_at_a_time.append(&[byte]);
        }
        let incremental_hash_one_at_a_time = hasher_one_at_a_time.finalize();
        assert_eq!(recursive_hash, incremental_hash_one_at_a_time);
    }
}

#[test]
fn test_zero_bytes() {
    let mut key = [0; KEY_BYTES];
    paint_test_input(&mut key);
    let key_words = words_from_key_bytes(&key);
    let mut state = iv(&key_words);
    let block = [0; BLOCK_BYTES];
    let flags = Flags::CHUNK_START | Flags::CHUNK_END | Flags::ROOT;
    portable::compress(&mut state, &block, 0, 0, flags.bits());
    let expected_hash = hash_from_state_words(&state);

    assert_eq!(expected_hash, hash_keyed(&[], &key));

    let hasher = Hasher::new_keyed(&key);
    assert_eq!(expected_hash, hasher.finalize());
}

#[test]
fn test_one_byte() {
    let mut key = [0; KEY_BYTES];
    paint_test_input(&mut key);
    let key_words = words_from_key_bytes(&key);
    let mut state = iv(&key_words);
    let mut block = [0; BLOCK_BYTES];
    block[0] = 9;
    let flags = Flags::CHUNK_START | Flags::CHUNK_END | Flags::ROOT;
    portable::compress(&mut state, &block, 1, 0, flags.bits());
    let expected_hash = hash_from_state_words(&state);

    assert_eq!(expected_hash, hash_keyed(&[9], &key));

    let mut hasher = Hasher::new_keyed(&key);
    hasher.append(&[9]);
    assert_eq!(expected_hash, hasher.finalize());
}

#[test]
fn test_three_blocks() {
    let mut input = [0; 2 * BLOCK_BYTES + 1];
    paint_test_input(&mut input);
    let mut key = [0; KEY_BYTES];
    paint_test_input(&mut key);
    let key_words = words_from_key_bytes(&key);
    let mut state = iv(&key_words);

    let block0 = array_ref!(input, 0, BLOCK_BYTES);
    portable::compress(
        &mut state,
        &block0,
        BLOCK_BYTES as Word,
        0,
        Flags::CHUNK_START.bits(),
    );

    let block1 = array_ref!(input, BLOCK_BYTES, BLOCK_BYTES);
    portable::compress(
        &mut state,
        &block1,
        BLOCK_BYTES as Word,
        0, // Subsequent blocks keep using the chunk's starting offset.
        Flags::empty().bits(),
    );

    let mut block2 = [0; BLOCK_BYTES];
    block2[0] = *input.last().unwrap();
    portable::compress(
        &mut state,
        &block2,
        1,
        0, // Subsequent blocks keep using the chunk's starting offset.
        (Flags::CHUNK_END | Flags::ROOT).bits(),
    );

    let expected_hash = hash_from_state_words(&state);

    assert_eq!(expected_hash, hash_keyed(&input, &key));

    let mut hasher = Hasher::new_keyed(&key);
    hasher.append(&input);
    assert_eq!(expected_hash, hasher.finalize());
}

#[test]
fn test_three_chunks() {
    let mut input = [0; 2 * CHUNK_BYTES + 1];
    paint_test_input(&mut input);
    let mut key = [0; KEY_BYTES];
    paint_test_input(&mut key);
    let key_words = words_from_key_bytes(&key);

    let chunk0 = hash_chunk(&input[..CHUNK_BYTES], &key_words, 0, IsRoot::NotRoot);
    let chunk1 = hash_chunk(
        &input[CHUNK_BYTES..][..CHUNK_BYTES],
        &key_words,
        CHUNK_BYTES as u64,
        IsRoot::NotRoot,
    );
    let chunk2 = hash_chunk(
        &input[2 * CHUNK_BYTES..],
        &key_words,
        2 * CHUNK_BYTES as u64,
        IsRoot::NotRoot,
    );

    let left_parent = hash_parent(&chunk0, &chunk1, &key_words, IsRoot::NotRoot);

    let root_parent = hash_parent(&left_parent, &chunk2, &key_words, IsRoot::Root);

    let expected_hash = hash_from_state_words(&root_parent);

    assert_eq!(expected_hash, hash_keyed(&input, &key));

    let mut hasher = Hasher::new_keyed(&key);
    hasher.append(&input);
    assert_eq!(expected_hash, hasher.finalize());
}

#[test]
fn test_default_key() {
    let default_key = &[0; KEY_BYTES];

    let expected_hash = hash_keyed(b"abc", &default_key);

    assert_eq!(expected_hash, hash(b"abc"));

    let mut hasher = Hasher::new();
    hasher.append(b"abc");
    assert_eq!(expected_hash, hasher.finalize());
}

#[test]
fn test_xof_small() {
    let input = b"abc";

    let mut hasher = Hasher::new();
    hasher.append(input);

    // Pin hasher as immutable for all of the following.
    let hasher = hasher;

    let expected_hash = hasher.finalize();
    let mut xof = hasher.finalize_xof();

    let first_bytes = xof.read();
    assert_eq!(expected_hash.as_bytes(), &first_bytes);

    let second_bytes = xof.read();
    assert!(first_bytes != second_bytes);

    let third_bytes = xof.read();
    assert!(first_bytes != third_bytes);
    assert!(second_bytes != third_bytes);
}
