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
    let mut input_buf = [0; 100 * CHUNK_LEN];
    paint_test_input(&mut input_buf);
    for &case in TEST_CASES {
        let input = &input_buf[..case];
        let key = array_ref!(input_buf, 0, KEY_LEN);

        let recursive_hash = hash_keyed(input, key);
        let recursive_kdf = kdf(key, input);

        let incremental_hash_all = Hasher::new_keyed(key).update(input).finalize();
        assert_eq!(recursive_hash, incremental_hash_all);

        // Private APIs currently.
        let kdf_all = Hasher::new_keyed_flags(key, Flags::KDF)
            .update(input)
            .finalize();
        assert_eq!(&recursive_kdf, kdf_all.as_bytes());

        let mut hasher_one_at_a_time = Hasher::new_keyed(key);
        for &byte in input {
            hasher_one_at_a_time.update(&[byte]);
        }
        assert_eq!(recursive_hash, hasher_one_at_a_time.finalize());

        // Private APIs currently.
        let mut kdf_one_at_a_time = Hasher::new_keyed_flags(key, Flags::KDF);
        for &byte in input {
            kdf_one_at_a_time.update(&[byte]);
        }
        assert_eq!(&recursive_kdf, kdf_one_at_a_time.finalize().as_bytes());
    }
}

#[test]
fn test_zero_bytes() {
    let mut key = [0; KEY_LEN];
    paint_test_input(&mut key);
    let key_words = words_from_key_bytes(&key);
    let mut state = iv(&key_words);
    let block = [0; BLOCK_LEN];
    let flags = Flags::CHUNK_START | Flags::CHUNK_END | Flags::ROOT;
    portable::compress(&mut state, &block, 0, 0, flags.bits());
    let expected_hash: Hash = bytes_from_state_words(&state).into();

    assert_eq!(expected_hash, hash_keyed(&[], &key));

    let hasher = Hasher::new_keyed(&key);
    assert_eq!(expected_hash, hasher.finalize());
}

#[test]
fn test_one_byte() {
    let mut key = [0; KEY_LEN];
    paint_test_input(&mut key);
    let key_words = words_from_key_bytes(&key);
    let mut state = iv(&key_words);
    let mut block = [0; BLOCK_LEN];
    block[0] = 9;
    let flags = Flags::CHUNK_START | Flags::CHUNK_END | Flags::ROOT;
    portable::compress(&mut state, &block, 1, 0, flags.bits());
    let expected_hash: Hash = bytes_from_state_words(&state).into();

    assert_eq!(expected_hash, hash_keyed(&[9], &key));

    let mut hasher = Hasher::new_keyed(&key);
    hasher.update(&[9]);
    assert_eq!(expected_hash, hasher.finalize());
}

type Construction = fn(&[u8], &[u8; KEY_LEN], Flags) -> Hash;

fn exercise_construction(construction: Construction, input_len: usize) {
    let mut input_buf = [0; 65536];
    paint_test_input(&mut input_buf);
    let input = &input_buf[..input_len];
    let key = array_ref!(input, 7, KEY_LEN);

    // Exercise the default hash.
    let expected_default_hash = construction(&input, &[0; KEY_LEN], Flags::empty());
    assert_eq!(expected_default_hash, hash(&input));
    assert_eq!(expected_default_hash, hash_keyed(&input, &[0; KEY_LEN]));
    assert_eq!(expected_default_hash, hash_xof(&input).read());
    assert_eq!(
        expected_default_hash,
        hash_keyed_xof(&input, &[0; KEY_LEN]).read(),
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
        Hasher::new().update(&input).finalize_xof().read(),
    );
    assert_eq!(
        expected_default_hash,
        Hasher::new_keyed(&[0; KEY_LEN])
            .update(&input)
            .finalize_xof()
            .read(),
    );

    // Exercise the keyed hash.
    let expected_keyed_hash = construction(&input, key, Flags::empty());
    assert_eq!(expected_keyed_hash, hash_keyed(&input, key));
    assert_eq!(expected_keyed_hash, hash_keyed_xof(&input, key).read(),);
    assert_eq!(
        expected_keyed_hash,
        Hasher::new_keyed(key).update(&input).finalize(),
    );
    assert_eq!(
        expected_keyed_hash,
        Hasher::new_keyed(key).update(&input).finalize_xof().read(),
    );

    // Exercise the KDF.
    let expected_kdf_out = *construction(&input, key, Flags::KDF).as_bytes();
    assert_eq!(expected_kdf_out, kdf(key, &input));
    assert_eq!(expected_kdf_out, kdf_xof(key, &input).read());
    // These are currently private APIs but it's nice to test them anyway.
    assert_eq!(
        &expected_kdf_out,
        Hasher::new_keyed_flags(key, Flags::KDF)
            .update(&input)
            .finalize()
            .as_bytes(),
    );
    assert_eq!(
        expected_kdf_out,
        Hasher::new_keyed_flags(key, Flags::KDF)
            .update(&input)
            .finalize_xof()
            .read(),
    );
}

fn three_blocks_construction(input_buf: &[u8], key: &[u8; KEY_LEN], flags: Flags) -> Hash {
    let key_words = words_from_key_bytes(&key);
    let mut state = iv(&key_words);

    let block0 = array_ref!(input_buf, 0, BLOCK_LEN);
    portable::compress(
        &mut state,
        &block0,
        BLOCK_LEN as Word,
        0,
        (flags | Flags::CHUNK_START).bits(),
    );

    let block1 = array_ref!(input_buf, BLOCK_LEN, BLOCK_LEN);
    portable::compress(
        &mut state,
        &block1,
        BLOCK_LEN as Word,
        0, // Subsequent blocks keep using the chunk's starting offset.
        flags.bits(),
    );

    let mut block2 = [0; BLOCK_LEN];
    block2[0] = input_buf[2 * BLOCK_LEN];
    portable::compress(
        &mut state,
        &block2,
        1,
        0, // Subsequent blocks keep using the chunk's starting offset.
        (flags | Flags::CHUNK_END | Flags::ROOT).bits(),
    );

    bytes_from_state_words(&state).into()
}

#[test]
fn test_three_blocks() {
    exercise_construction(three_blocks_construction, 2 * BLOCK_LEN + 1);
}

fn three_chunks_construction(input_buf: &[u8], key: &[u8; KEY_LEN], flags: Flags) -> Hash {
    let key_words = words_from_key_bytes(&key);
    let chunk0 = hash_one_chunk(
        &input_buf[..CHUNK_LEN],
        &key_words,
        0,
        flags,
        IsRoot::NotRoot,
        Platform::detect(),
    )
    .read();
    let chunk1 = hash_one_chunk(
        &input_buf[CHUNK_LEN..][..CHUNK_LEN],
        &key_words,
        CHUNK_LEN as u64,
        flags,
        IsRoot::NotRoot,
        Platform::detect(),
    )
    .read();
    let chunk2 = hash_one_chunk(
        &input_buf[2 * CHUNK_LEN..][..1],
        &key_words,
        2 * CHUNK_LEN as u64,
        flags,
        IsRoot::NotRoot,
        Platform::detect(),
    )
    .read();
    let left_parent = hash_one_parent(
        &chunk0,
        &chunk1,
        &key_words,
        flags,
        IsRoot::NotRoot,
        Platform::detect(),
    )
    .read();
    hash_one_parent(
        &left_parent,
        &chunk2,
        &key_words,
        flags,
        IsRoot::Root,
        Platform::detect(),
    )
    .read()
    .into()
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
    let expected_hash = hash(input);

    let mut hasher = Hasher::new();
    hasher.update(input);
    // Pin hasher as immutable for all of the following.
    let hasher = hasher;

    assert_eq!(expected_hash, hasher.finalize());

    let mut xof = hasher.finalize_xof();
    let first_bytes = xof.read();
    assert_eq!(expected_hash.as_bytes(), &first_bytes);

    let second_bytes = xof.read();
    assert!(first_bytes != second_bytes);

    let third_bytes = xof.read();
    assert!(first_bytes != third_bytes);
    assert!(second_bytes != third_bytes);
}

#[test]
fn test_keyed_xof_wrappers() {
    let input = b"abc";
    let expected_hash = hash(input);
    let mut xof = hash_xof(input);
    assert!(expected_hash == xof.read());

    let key = [42; KEY_LEN];
    let expected_keyed_hash = hash_keyed(input, &key);
    let mut keyed_xof = hash_keyed_xof(input, &key);
    assert_eq!(expected_keyed_hash, keyed_xof.read());
}
