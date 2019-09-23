// NB: This is only for benchmarking. The guy who wrote this file hasn't
// touched C since college. Please don't use this code in production.

#include <assert.h>
#include <stdbool.h>
#include <string.h>

#include "baokeshed_impl.h"

typedef struct {
  uint32_t state[8];
  uint64_t offset;
  uint16_t count;
  uint8_t buf[BLOCK_LEN];
  uint8_t buf_len;
} chunk_state;

void chunk_state_init(chunk_state *self, const uint32_t key_words[8],
                      uint64_t offset) {
  init_iv(key_words, self->state);
  self->offset = offset;
  self->count = 0;
  self->buf_len = 0;
  memset(self->buf, 0, BLOCK_LEN);
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

void chunk_state_update(chunk_state *self, const uint8_t *input,
                        size_t input_len, uint32_t context) {
  if (self->buf_len > 0) {
    size_t take = chunk_state_fill_buf(self, input, input_len);
    input += take;
    input_len -= take;
    if (input_len > 0) {
      compress(self->state, self->buf, BLOCK_LEN, self->offset,
               chunk_state_maybe_start_flag(self), context);
      self->count += (uint16_t)BLOCK_LEN;
      self->buf_len = 0;
      memset(self->buf, 0, BLOCK_LEN);
    }
  }

  while (input_len > BLOCK_LEN) {
    compress(self->state, input, BLOCK_LEN, self->offset,
             chunk_state_maybe_start_flag(self), context);
    self->count += (uint16_t)BLOCK_LEN;
    input += BLOCK_LEN;
    input_len -= BLOCK_LEN;
  }

  size_t take = chunk_state_fill_buf(self, input, input_len);
  input += take;
  input_len -= take;
}

void chunk_state_finalize(const chunk_state *self, uint32_t context,
                          bool is_root, uint8_t out[OUT_LEN]) {
  uint32_t state_copy[8];
  memcpy(state_copy, self->state, sizeof(state_copy));
  uint8_t flags = chunk_state_maybe_start_flag(self) | CHUNK_END;
  if (is_root) {
    flags |= ROOT;
  }
  compress(state_copy, self->buf, self->buf_len, self->offset, flags, context);
  write_state_bytes(state_copy, out);
}

INLINE void hash_one_parent(const uint8_t block[BLOCK_LEN],
                            const uint32_t key_words[8], bool is_root,
                            uint32_t context, uint8_t out[OUT_LEN]) {
  uint8_t flags = PARENT;
  if (is_root) {
    flags |= ROOT;
  }
  uint32_t state[8];
  init_iv(key_words, state);
  compress(state, block, BLOCK_LEN, 0, flags, context);
  write_state_bytes(state, out);
}

typedef struct {
  chunk_state chunk;
  uint32_t key_words[8];
  uint32_t context;
  uint8_t subtree_hashes_len;
  uint8_t subtree_hashes[MAX_DEPTH * OUT_LEN];
} hasher;

void hasher_init(hasher *self, const uint8_t key[KEY_LEN], uint32_t context) {
  load_key_words(key, self->key_words);
  chunk_state_init(&self->chunk, self->key_words, 0);
  self->context = context;
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
  hash_one_parent(&self->subtree_hashes[parent_block_start], self->key_words,
                  false, self->context, out);
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

void hasher_update(hasher *self, const void *input, size_t input_len) {
  uint8_t *input_bytes = (uint8_t *)input;
  while (input_len > 0) {
    if (chunk_state_len(&self->chunk) == CHUNK_LEN) {
      uint8_t hash[OUT_LEN];
      chunk_state_finalize(&self->chunk, self->context, false, hash);
      hasher_push_chunk_hash(self, hash, self->chunk.offset);
      uint64_t new_offset = self->chunk.offset + CHUNK_LEN;
      chunk_state_init(&self->chunk, self->key_words, new_offset);
      // Work ahead, to simplify finalize().
      while (hasher_needs_merge(self, new_offset)) {
        hasher_merge_parent(self);
      }
    }
    size_t take = CHUNK_LEN - chunk_state_len(&self->chunk);
    if (take > input_len) {
      take = input_len;
    }
    chunk_state_update(&self->chunk, input_bytes, take, self->context);
    input_bytes += take;
    input_len -= take;
  }
}

void hasher_finalize(const hasher *self, uint8_t out[OUT_LEN]) {
  if (self->subtree_hashes_len == 0) {
    chunk_state_finalize(&self->chunk, self->context, true, out);
    return;
  }
  uint8_t working_hash[OUT_LEN];
  chunk_state_finalize(&self->chunk, self->context, false, working_hash);
  size_t next_subtree_start =
      (((size_t)self->subtree_hashes_len) - 1) * OUT_LEN;
  while (true) {
    uint8_t block[BLOCK_LEN];
    memcpy(block, &self->subtree_hashes[next_subtree_start], OUT_LEN);
    memcpy(&block[OUT_LEN], working_hash, OUT_LEN);
    if (next_subtree_start == 0) {
      hash_one_parent(block, self->key_words, true, self->context, out);
      return;
    }
    hash_one_parent(block, self->key_words, false, self->context, working_hash);
    next_subtree_start -= OUT_LEN;
  }
}
