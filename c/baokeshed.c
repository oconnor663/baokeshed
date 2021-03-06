// NB: This is only for benchmarking. The guy who wrote this file hasn't
// touched C since college. Please don't use this code in production.

#include <assert.h>
#include <stdbool.h>
#include <string.h>

#include "baokeshed_impl.h"

typedef struct {
  uint32_t state[8];
  uint32_t key[8];
  uint64_t offset;
  uint16_t count;
  uint8_t buf[BLOCK_LEN];
  uint8_t buf_len;
  uint8_t flags;
} chunk_state;

// Non-inline for unit testing.
void chunk_state_init(chunk_state *self, const uint32_t key[8], uint8_t flags) {
  memcpy(self->state, key, KEY_LEN);
  memcpy(self->key, key, KEY_LEN);
  self->offset = 0;
  self->count = 0;
  memset(self->buf, 0, BLOCK_LEN);
  self->buf_len = 0;
  self->flags = flags;
}

void chunk_state_reset(chunk_state *self, uint64_t new_offset) {
  memcpy(self->state, self->key, KEY_LEN);
  self->offset = new_offset;
  self->count = 0;
  memset(self->buf, 0, BLOCK_LEN);
  self->buf_len = 0;
}

INLINE size_t chunk_state_len(const chunk_state *self) {
  return ((size_t)self->count) + ((size_t)self->buf_len);
}

INLINE size_t chunk_state_fill_buf(chunk_state *self, const uint8_t *input,
                                   size_t input_len) {
  size_t take = BLOCK_LEN - ((size_t)self->buf_len);
  if (take > input_len) {
    take = input_len;
  }
  uint8_t *dest = self->buf + ((size_t)self->buf_len);
  memcpy(dest, input, take);
  self->buf_len += (uint8_t)take;
  return take;
}

INLINE uint8_t chunk_state_maybe_start_flag(const chunk_state *self) {
  if (self->count == 0) {
    return CHUNK_START;
  } else {
    return 0;
  }
}

INLINE void compress(uint32_t state[8], const uint8_t block[BLOCK_LEN],
                     uint8_t block_len, uint64_t offset, uint8_t flags) {
#if BAOKESHED_USE_AVX512
  compress_avx512(state, block, block_len, offset, flags);
#elif BAOKESHED_USE_SSE41
  compress_sse41(state, block, block_len, offset, flags);
#else
  compress_portable(state, block, block_len, offset, flags);
#endif
}

// Non-inline for unit testing.
void chunk_state_update(chunk_state *self, const uint8_t *input,
                        size_t input_len) {
  if (self->buf_len > 0) {
    size_t take = chunk_state_fill_buf(self, input, input_len);
    input += take;
    input_len -= take;
    if (input_len > 0) {
      compress(self->state, self->buf, BLOCK_LEN, self->offset,
               self->flags | chunk_state_maybe_start_flag(self));
      self->count += (uint16_t)BLOCK_LEN;
      self->buf_len = 0;
      memset(self->buf, 0, BLOCK_LEN);
    }
  }

  while (input_len > BLOCK_LEN) {
    compress(self->state, input, BLOCK_LEN, self->offset,
             self->flags | chunk_state_maybe_start_flag(self));
    self->count += (uint16_t)BLOCK_LEN;
    input += BLOCK_LEN;
    input_len -= BLOCK_LEN;
  }

  size_t take = chunk_state_fill_buf(self, input, input_len);
  input += take;
  input_len -= take;
}

// Non-inline for unit testing.
void chunk_state_finalize(const chunk_state *self, bool is_root,
                          uint8_t out[OUT_LEN]) {
  uint32_t state_copy[8];
  memcpy(state_copy, self->state, sizeof(state_copy));
  uint8_t block_flags =
      self->flags | chunk_state_maybe_start_flag(self) | CHUNK_END;
  if (is_root) {
    block_flags |= ROOT;
  }
  compress(state_copy, self->buf, self->buf_len, self->offset, block_flags);
  write_state_bytes(state_copy, out);
}

INLINE void hash_one_parent(const uint8_t block[BLOCK_LEN],
                            const uint32_t key[8], bool is_root, uint8_t flags,
                            uint8_t out[OUT_LEN]) {
  uint8_t block_flags = flags | PARENT;
  if (is_root) {
    block_flags |= ROOT;
  }
  uint32_t state[8];
  memcpy(state, key, KEY_LEN);
  compress(state, block, BLOCK_LEN, 0, block_flags);
  write_state_bytes(state, out);
}

typedef struct {
  chunk_state chunk;
  uint8_t subtree_hashes_len;
  uint8_t subtree_hashes[MAX_DEPTH * OUT_LEN];
} hasher;

void hasher_init(hasher *self, const uint8_t key[KEY_LEN], uint8_t flags) {
  uint32_t key_words[8];
  load_key_words(key, key_words); // This handles big-endianness.
  chunk_state_init(&self->chunk, key_words, flags);
  self->subtree_hashes_len = 0;
}

INLINE bool hasher_needs_merge(const hasher *self, uint64_t total_bytes) {
  uint64_t total_chunks = total_bytes / CHUNK_LEN;
  return self->subtree_hashes_len > popcnt(total_chunks);
}

INLINE void hasher_merge_parent(hasher *self) {
  size_t parent_block_start =
      (((size_t)self->subtree_hashes_len) - 2) * OUT_LEN;
  uint8_t out[OUT_LEN];
  hash_one_parent(&self->subtree_hashes[parent_block_start], self->chunk.key,
                  false, self->chunk.flags, out);
  memcpy(&self->subtree_hashes[parent_block_start], out, OUT_LEN);
  self->subtree_hashes_len -= 1;
}

INLINE void hasher_push_chunk_hash(hasher *self, uint8_t hash[OUT_LEN],
                                   uint64_t offset) {
  assert(self->subtree_hashes_len < MAX_DEPTH);
  while (hasher_needs_merge(self, offset)) {
    hasher_merge_parent(self);
  }
  memcpy(&self->subtree_hashes[self->subtree_hashes_len * OUT_LEN], hash,
         OUT_LEN);
  self->subtree_hashes_len += 1;
}

INLINE void hash_many(const uint8_t *const *inputs, size_t num_inputs,
                      size_t blocks, const uint32_t key_words[8],
                      uint64_t offset, const uint64_t offset_deltas[17],
                      uint8_t flags, uint8_t flags_start, uint8_t flags_end,
                      uint8_t *out) {
#if BAOKESHED_USE_AVX512
  hash_many_avx512(inputs, num_inputs, blocks, key_words, offset, offset_deltas,
                   flags, flags_start, flags_end, out);
#elif BAOKESHED_USE_AVX2
  hash_many_avx2(inputs, num_inputs, blocks, key_words, offset, offset_deltas,
                 flags, flags_start, flags_end, out);
#elif BAOKESHED_USE_SSE41
  hash_many_sse41(inputs, num_inputs, blocks, key_words, offset, offset_deltas,
                  flags, flags_start, flags_end, out);
#elif BAOKESHED_USE_NEON
  hash_many_neon(inputs, num_inputs, blocks, key_words, offset, offset_deltas,
                 flags, flags_start, flags_end, out);
#else
  hash_many_portable(inputs, num_inputs, blocks, key_words, offset,
                     offset_deltas, flags, flags_start, flags_end, out);
#endif
}

void hasher_update(hasher *self, const void *input, size_t input_len) {
  const uint8_t *input_bytes = (const uint8_t *)input;

  // If we already have a partial chunk, or if this is the very first chunk
  // (and it could be the root), we need to add bytes to the chunk state.
  bool is_first_chunk = self->chunk.offset == 0;
  bool maybe_root = is_first_chunk && input_len == CHUNK_LEN;
  if (maybe_root || chunk_state_len(&self->chunk) > 0) {
    size_t take = CHUNK_LEN - chunk_state_len(&self->chunk);
    if (take > input_len) {
      take = input_len;
    }
    chunk_state_update(&self->chunk, input_bytes, take);
    input_bytes += take;
    input_len -= take;
    // If we've filled the current chunk and there's more coming, finalize this
    // chunk and proceed. In this case we know it's not the root.
    if (input_len > 0) {
      uint8_t out[OUT_LEN];
      chunk_state_finalize(&self->chunk, false, out);
      hasher_push_chunk_hash(self, out, self->chunk.offset);
      chunk_state_reset(&self->chunk, self->chunk.offset + CHUNK_LEN);
    } else {
      return;
    }
  }

  // Hash as many whole chunks as we can, without buffering anything. At this
  // point we know none of them can be the root.
  uint8_t out[OUT_LEN * MAX_SIMD_DEGREE];
  const uint8_t *chunks[MAX_SIMD_DEGREE];
  size_t num_chunks = 0;
  while (input_len >= CHUNK_LEN) {
    while (input_len >= CHUNK_LEN && num_chunks < MAX_SIMD_DEGREE) {
      chunks[num_chunks] = input_bytes;
      input_bytes += CHUNK_LEN;
      input_len -= CHUNK_LEN;
      num_chunks += 1;
    }
    hash_many(chunks, num_chunks, CHUNK_LEN / BLOCK_LEN, self->chunk.key,
              self->chunk.offset, CHUNK_OFFSET_DELTAS, self->chunk.flags,
              CHUNK_START, CHUNK_END, out);
    for (size_t chunk_index = 0; chunk_index < num_chunks; chunk_index++) {
      // The chunk state is empty here, but it stores the offset of the next
      // chunk hash we need to push. Use that offset, and then move it forward.
      hasher_push_chunk_hash(self, &out[chunk_index * OUT_LEN],
                             self->chunk.offset);
      self->chunk.offset += CHUNK_LEN;
    }
    num_chunks = 0;
  }

  // If there's any remaining input less than a full chunk, add it to the chunk
  // state. In that case, also do a final merge loop to make sure the subtree
  // stack doesn't contain any unmerged pairs. The remaining input means we
  // know these merges are non-root. This merge loop isn't strictly necessary
  // here, because hasher_push_chunk_hash already does its own merge loop, but
  // it simplifies hasher_finalize below.
  if (input_len > 0) {
    while (hasher_needs_merge(self, self->chunk.offset)) {
      hasher_merge_parent(self);
    }
    chunk_state_update(&self->chunk, input_bytes, input_len);
  }
}

void hasher_finalize(const hasher *self, uint8_t out[OUT_LEN]) {
  // If the subtree stack is empty, then the current chunk is the root.
  if (self->subtree_hashes_len == 0) {
    chunk_state_finalize(&self->chunk, true, out);
    return;
  }
  // If there are any bytes in the chunk state, finalize that chunk and do a
  // roll-up merge between that chunk hash and every subtree in the stack. In
  // this case, the extra merge loop at the end of hasher_update guarantees
  // that none of the subtrees in the stack need to be merged with each other
  // first. Otherwise, if there are no bytes in the chunk state, then the top
  // of the stack is a chunk hash, and we start the merge from that.
  uint8_t working_hash[OUT_LEN];
  size_t next_subtree_start;
  if (chunk_state_len(&self->chunk) > 0) {
    chunk_state_finalize(&self->chunk, false, working_hash);
    next_subtree_start = (self->subtree_hashes_len - 1) * OUT_LEN;
  } else {
    size_t last_hash_start = (self->subtree_hashes_len - 1) * OUT_LEN;
    memcpy(working_hash, &self->subtree_hashes[last_hash_start], OUT_LEN);
    next_subtree_start = (self->subtree_hashes_len - 2) * OUT_LEN;
  }
  while (true) {
    bool is_root = (next_subtree_start == 0);
    uint8_t block[BLOCK_LEN];
    memcpy(block, &self->subtree_hashes[next_subtree_start], OUT_LEN);
    memcpy(&block[OUT_LEN], working_hash, OUT_LEN);
    hash_one_parent(block, self->chunk.key, is_root, self->chunk.flags,
                    working_hash);
    if (is_root) {
      memcpy(out, working_hash, OUT_LEN);
      return;
    }
    next_subtree_start -= OUT_LEN;
  }
}
