/*
 * Lab03 verify - 自动验证程序
 *
 * 4 个 [TEST] 必须全部 [PASS].
 *
 * 关键约束: 本文件只调用 student.c 暴露的 2 个函数 + 公开的
 * framework API (embedding_create / embedding_free /
 * embedding_init_random / embedding_init_sinusoidal_position).
 *
 * 不读 Embedding 结构体内部字段 (除 emb->vocab_size /
 * emb->max_seq_len / emb->hidden_dim 之外, 不读 token_embedding 等
 * 内部 Tensor). 所有"输入"从函数参数拿, 不依赖全局状态.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "student.h"
#include "embedding.h"

/* ---------- 测试工具宏 ---------- */
#define TEST_BEGIN(n)                                                       \
    do {                                                                     \
        printf("[TEST %d] ", (n));                                           \
        fflush(stdout);                                                      \
    } while (0)
#define TEST_PASS()                                                          \
    do {                                                                     \
        printf(" ... [PASS]\n");                                             \
    } while (0)
#define TEST_FAIL(fmt, ...)                                                  \
    do {                                                                     \
        printf(" ... [FAIL] " fmt "\n", ##__VA_ARGS__);                      \
        g_failures++;                                                        \
    } while (0)

static int g_failures = 0;     /* 失败计数 (模块内静态, 不是 framework 状态) */

#define VOCAB 260
#define HD    64
#define MAXLEN 128

/* ---------- TEST 1: PE(pos=0) 偶数维=0, 奇数维=1 ---------- */
static void test_pe_at_zero(void) {
    TEST_BEGIN(1);

    /* 抽样 8 个偶数维, 8 个奇数维, 都应该精确是 0 / 1 */
    int ok = 1;
    for (int d = 0; d < 8; d++) {
        if (fabsf(student_pe_sinusoidal(0, 2 * d,     HD) - 0.0f) > 1e-5f) ok = 0;
        if (fabsf(student_pe_sinusoidal(0, 2 * d + 1, HD) - 1.0f) > 1e-5f) ok = 0;
    }
    if (!ok) {
        TEST_FAIL("PE(0, 偶数)!=0 或 PE(0, 奇数)!=1");
    } else {
        TEST_PASS();
    }
}

/* ---------- TEST 2: PE(pos=1, dim=0) 接近 sin(1) ---------- */
static void test_pe_at_one(void) {
    TEST_BEGIN(2);

    /* PE(1, 0) = sin(1 / 10000^0) = sin(1) ≈ 0.8415 */
    float v0 = student_pe_sinusoidal(1, 0, HD);
    float expected0 = sinf(1.0f);
    if (fabsf(v0 - expected0) > 1e-4f) {
        TEST_FAIL("PE(1,0)=%f, want sin(1)=%f", v0, expected0);
        return;
    }

    /* PE(1, 1) = cos(1) ≈ 0.5403 */
    float v1 = student_pe_sinusoidal(1, 1, HD);
    float expected1 = cosf(1.0f);
    if (fabsf(v1 - expected1) > 1e-4f) {
        TEST_FAIL("PE(1,1)=%f, want cos(1)=%f", v1, expected1);
        return;
    }

    TEST_PASS();
}

/* ---------- TEST 3: 相同 token 不同位置 -> 输出不同 ---------- */
static void test_forward_same_token_diff_pos(void) {
    TEST_BEGIN(3);

    Embedding* emb = embedding_create(VOCAB, HD, MAXLEN);
    if (emb == NULL) { TEST_FAIL("embedding_create returned NULL"); return; }
    embedding_init_random(emb, 0.02f);
    embedding_init_sinusoidal_position(emb);

    /* 3 个相同的 'H' (ASCII 76),  pos = 0, 1, 2 */
    int ids[] = {76, 76, 76};
    int seq_len = 3;
    int out_shape[] = {seq_len, HD};
    Tensor* out = tensor_zeros(2, out_shape);
    if (out == NULL) { TEST_FAIL("tensor_zeros returned NULL"); embedding_free(emb); return; }

    student_embedding_forward(emb, ids, seq_len, out);

    /* 算 out[0] 与 out[2] 的 L2 距离平方; 应 > 0 */
    float diff2 = 0.0f;
    for (int d = 0; d < HD; d++) {
        float delta = out->data[0 * HD + d] - out->data[2 * HD + d];
        diff2 += delta * delta;
    }

    if (diff2 <= 0.0f) {
        TEST_FAIL("相同 token 不同位置 -> diff^2 = %f (应 > 0)", diff2);
    } else {
        TEST_PASS();
    }

    tensor_free(out);
    embedding_free(emb);
}

/* ---------- TEST 4: 不同 token 相同位置 -> 输出不同 ---------- */
static void test_forward_diff_token_same_pos(void) {
    TEST_BEGIN(4);

    Embedding* emb = embedding_create(VOCAB, HD, MAXLEN);
    if (emb == NULL) { TEST_FAIL("embedding_create returned NULL"); return; }
    embedding_init_random(emb, 0.02f);
    embedding_init_sinusoidal_position(emb);

    /* 不同字符 'H' (76) 和 'i' (109), 同一 pos=0 */
    int ids[] = {76, 109};
    int seq_len = 2;
    int out_shape[] = {seq_len, HD};
    Tensor* out = tensor_zeros(2, out_shape);
    if (out == NULL) { TEST_FAIL("tensor_zeros returned NULL"); embedding_free(emb); return; }

    student_embedding_forward(emb, ids, seq_len, out);

    float diff2 = 0.0f;
    for (int d = 0; d < HD; d++) {
        float delta = out->data[0 * HD + d] - out->data[1 * HD + d];
        diff2 += delta * delta;
    }

    if (diff2 <= 0.0f) {
        TEST_FAIL("不同 token 相同位置 -> diff^2 = %f (应 > 0)", diff2);
    } else {
        TEST_PASS();
    }

    tensor_free(out);
    embedding_free(emb);
}

/* ---------- 入口 ---------- */
int verify_run_all(void) {
    g_failures = 0;
    printf("========================================\n");
    printf("       Lab03 - 词嵌入与位置编码 验证     \n");
    printf("========================================\n");

    test_pe_at_zero();
    test_pe_at_one();
    test_forward_same_token_diff_pos();
    test_forward_diff_token_same_pos();

    printf("========================================\n");
    if (g_failures == 0) {
        printf("All tests passed!\n");
        return 0;
    } else {
        printf("%d test(s) FAILED.\n", g_failures);
        return 1;
    }
}
