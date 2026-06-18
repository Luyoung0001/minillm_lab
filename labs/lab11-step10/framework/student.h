#ifndef STUDENT_H
#define STUDENT_H

#include "kv_cache.h"

/* 申请一个 KVCache，按 (num_layers, max_seq_len, hidden_dim) 分配内存 */
KVCache* student_kv_cache_alloc(int num_layers, int max_seq_len, int hidden_dim);

/* 把 k_row / v_row（长度 = hidden_dim）写到 cache 第 layer_idx 层的第 pos 个位置 */
int student_kv_cache_append(KVCache* cache, int layer_idx, int pos,
                           const float* k_row, const float* v_row);

/* 把 cache->current_len 写到 *out_len，返回 0 成功 / -1 失败 */
int student_kv_cache_len(const KVCache* cache, int* out_len);

#endif
