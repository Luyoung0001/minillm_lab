#include "student.h"
#include "verify.h"
#include "kv_cache.h"
#include <stddef.h>

/* === 你需要实现的三个函数（每个 5~20 行） === */

KVCache* student_kv_cache_alloc(int num_layers, int max_seq_len, int hidden_dim) {
    /* TODO(student):
     *   1) 调 framework 的 kv_cache_create(...) 返回一个非空 cache
     *   2) 返回前可加一句 assert-style 检查（不推荐用 stdlib assert；
     *      课程约定遇到 NULL 直接 return NULL 即可）
     *   不要自己写 malloc 分配 data——framework 会做。
     */
    (void)num_layers; (void)max_seq_len; (void)hidden_dim;
    return NULL;  /* placeholder */
}

int student_kv_cache_append(KVCache* cache, int layer_idx, int pos,
                            const float* k_row, const float* v_row) {
    /* TODO(student):
     *   1) 检查 cache / k_row / v_row 非空
     *   2) 调 framework 的 kv_cache_update_pos(cache, layer_idx, pos,
     *                                          (float*)k_row, (float*)v_row);
     *   3) 调 kv_cache_set_len(cache, pos + 1) 让 current_len 跟上
     *   4) 返回 0；任意一步失败返回 -1
     *   不要自己 memcpy 到 cache->layers[..].k_cache->data——
     *   framework 的 update_pos 已经按 stride 写好了。
     */
    (void)cache; (void)layer_idx; (void)pos; (void)k_row; (void)v_row;
    return 0;  /* placeholder */
}

int student_kv_cache_len(const KVCache* cache, int* out_len) {
    /* TODO(student):
     *   检查 cache 和 out_len 非空，把 cache->current_len 赋给 *out_len
     *   返回 0。
     */
    (void)cache; (void)out_len;
    return 0;  /* placeholder */
}

int main(void) {
    return verify_run_all();
}
