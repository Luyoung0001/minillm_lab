#include "verify.h"
#include "student.h"
#include "kv_cache.h"
#include "tensor.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* ====== helpers（verify 和 student 都用：保证 verify.c 不靠 framework 内部状态） ====== */

/* 构造一个长度为 hidden_dim、值 = 0.1*i 的向量 */
static void make_row(float* dst, int hidden_dim, float scale) {
    for (int i = 0; i < hidden_dim; i++) dst[i] = scale * (float)(i + 1);
}

/* 比对两个 float 数组是否在 eps 容差内相等 */
static int close_enough(const float* a, const float* b, int n, float eps) {
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i];
        if (d < 0) d = -d;
        if (d > eps) return 0;
    }
    return 1;
}

/* ====== tests ====== */

static int test_alloc_returns_non_null(void) {
    KVCache* c = student_kv_cache_alloc(4, 16, 8);
    if (!c) return 0;
    kv_cache_free(c);
    return 1;
}

static int test_memory_size_matches_formula(void) {
    int L = 4, S = 16, H = 8;
    KVCache* c = student_kv_cache_alloc(L, S, H);
    if (!c) return 0;
    size_t got = kv_cache_memory_size(c);
    size_t expected = (size_t)2 * L * S * H * sizeof(float);
    kv_cache_free(c);
    return got == expected;
}

static int test_append_writes_back_same_data(void) {
    int L = 2, S = 8, H = 4;
    KVCache* c = student_kv_cache_alloc(L, S, H);
    if (!c) return 0;

    float k[H], v[H];
    make_row(k, H, 0.10f);  /* 0.10, 0.20, 0.30, 0.40 */
    make_row(v, H, 0.50f);  /* 0.50, 1.00, 1.50, 2.00 */

    if (student_kv_cache_append(c, /*layer=*/1, /*pos=*/3, k, v) != 0) {
        kv_cache_free(c);
        return 0;
    }

    /* 读回 layer 1 的 K cache 的 [3, :] 行 */
    Tensor* k_cache = kv_cache_get_k(c, 1);
    if (!k_cache || k_cache->ndim != 2 || k_cache->shape[0] != S || k_cache->shape[1] != H) {
        kv_cache_free(c);
        return 0;
    }
    /* row-major: 行 3 起点 = 3 * H */
    const float* row = k_cache->data + 3 * H;
    int ok_k = close_enough(row, k, H, 1e-5f);

    Tensor* v_cache = kv_cache_get_v(c, 1);
    const float* vrow = v_cache->data + 3 * H;
    int ok_v = close_enough(vrow, v, H, 1e-5f);

    kv_cache_free(c);
    return ok_k && ok_v;
}

static int test_len_reflects_appended_count(void) {
    int L = 1, S = 8, H = 2;
    KVCache* c = student_kv_cache_alloc(L, S, H);
    if (!c) return 0;

    float k[H], v[H];
    make_row(k, H, 1.0f);
    make_row(v, H, 1.0f);

    /* append 三个位置 */
    int rc = 0;
    rc |= student_kv_cache_append(c, 0, 0, k, v);
    rc |= student_kv_cache_append(c, 0, 1, k, v);
    rc |= student_kv_cache_append(c, 0, 2, k, v);
    if (rc != 0) { kv_cache_free(c); return 0; }

    int len = -1;
    if (student_kv_cache_len(c, &len) != 0) { kv_cache_free(c); return 0; }

    kv_cache_free(c);
    /* current_len 应该等于 3（pos+1 = 3） */
    return len == 3;
}

/* ====== runner ====== */

#define RUN(t, name) do { \
    int _ok = (t)(); \
    printf("[TEST %d] %-45s ... [%s]\n", \
           g_test_idx++, (name), _ok ? "PASS" : "FAIL"); \
    if (!_ok) g_fail = 1; \
} while (0)

static int g_test_idx = 1;
static int g_fail = 0;

int verify_run_all(void) {
    g_test_idx = 1;
    g_fail = 0;
    RUN(test_alloc_returns_non_null,       "kv_cache_alloc returns non-NULL");
    RUN(test_memory_size_matches_formula,  "kv_cache memory size matches formula");
    RUN(test_append_writes_back_same_data, "kv_cache_append writes back same data");
    RUN(test_len_reflects_appended_count,  "kv_cache_len reflects appended count");
    if (!g_fail) printf("[ALL PASS]\n");
    else         printf("[HAS FAIL]\n");
    return g_fail;
}
