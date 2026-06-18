#include "verify.h"
#include "student.h"

#include "tensor.h"
#include "math_ops.h"
#include "generate.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============== 测试计数 ============== */
static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do {                                              \
    if (cond) {                                                            \
        printf("  [PASS] %s\n", (msg));                                    \
        g_pass++;                                                          \
    } else {                                                               \
        printf("  [FAIL] %s (line %d)\n", (msg), __LINE__);                \
        g_fail++;                                                          \
    }                                                                      \
} while (0)

/* ============== helper ============== */

/* 创建一个 vocab_size 维的 logits 张量并填入给定数据 */
static Tensor* make_logits(int vocab_size, const float* data) {
    int shape[] = {vocab_size};
    Tensor* t = tensor_zeros(1, shape);
    if (data != NULL) {
        for (int i = 0; i < vocab_size; i++) {
            t->data[i] = data[i];
        }
    }
    return t;
}

/* 创建一个 vocab_size 维的全零 logits 张量 */
static Tensor* make_zeros(int vocab_size) {
    int shape[] = {vocab_size};
    return tensor_zeros(1, shape);
}

/* 创建一个 vocab_size 维的全 a 张量 */
static Tensor* make_full(int vocab_size, float a) {
    int shape[] = {vocab_size};
    Tensor* t = tensor_zeros(1, shape);
    for (int i = 0; i < vocab_size; i++) t->data[i] = a;
    return t;
}

/* 重复抽样 student_sample_temperature，返回每个索引被抽中的次数 [vocab_size] */
static void hist_temperature(Tensor* logits, float temperature, int n_samples, int* hist, int vocab_size) {
    for (int i = 0; i < vocab_size; i++) hist[i] = 0;
    for (int s = 0; s < n_samples; s++) {
        int idx = student_sample_temperature(logits, temperature);
        if (idx >= 0 && idx < vocab_size) hist[idx]++;
    }
}

/* 重复抽样 student_sample_top_k，返回每个索引被抽中的次数 */
static void hist_top_k(Tensor* logits, int k, float temperature, int n_samples, int* hist, int vocab_size) {
    for (int i = 0; i < vocab_size; i++) hist[i] = 0;
    for (int s = 0; s < n_samples; s++) {
        int idx = student_sample_top_k(logits, k, temperature);
        if (idx >= 0 && idx < vocab_size) hist[idx]++;
    }
}

/* ============== TEST 1: student_sample_greedy ============== */
static void test_student_sample_greedy(void) {
    printf("\n=== TEST 1: student_sample_greedy ===\n");

    /* 1a. 简单情形：最大 logit 在位置 3 */
    float data1[] = {1.0f, 2.0f, 5.0f, 7.0f, 3.0f};
    Tensor* logits = make_logits(5, data1);
    int idx = student_sample_greedy(logits);
    CHECK(idx == 3, "argmax picks position of max logit");
    tensor_free(logits);

    /* 1b. 平局时取第一个 */
    float data2[] = {4.0f, 4.0f, 4.0f};
    Tensor* logits2 = make_logits(3, data2);
    int idx2 = student_sample_greedy(logits2);
    CHECK(idx2 == 0, "argmax handles tied values (first wins)");
    tensor_free(logits2);

    /* 1c. 均匀分布（应仍返回某个位置，但要不崩） */
    float data3[] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    Tensor* logits3 = make_logits(7, data3);
    int idx3 = student_sample_greedy(logits3);
    CHECK(idx3 >= 0 && idx3 < 7, "argmax works on a uniform distribution");
    tensor_free(logits3);

    /* 1d. logits 没被破坏（采样函数应当是 const-like 行为） */
    float data4[] = {1.0f, 3.0f, 2.0f};
    Tensor* logits4 = make_logits(3, data4);
    (void)student_sample_greedy(logits4);
    CHECK(logits4->data[0] == 1.0f && logits4->data[1] == 3.0f && logits4->data[2] == 2.0f,
          "logits not modified after greedy");
    tensor_free(logits4);
}

/* ============== TEST 2: student_sample_temperature ============== */
static void test_student_sample_temperature(void) {
    printf("\n=== TEST 2: student_sample_temperature ===\n");

    /* 2a. T=0.1 几乎总是返回最大 logit */
    {
        float data[] = {0.0f, 0.0f, 0.0f, 5.0f, 0.0f};
        Tensor* logits = make_logits(5, data);
        int hist[5] = {0};
        srand(42);
        hist_temperature(logits, 0.1f, 1000, hist, 5);
        /* 位置 3 应该几乎独占 */
        CHECK(hist[3] >= 990, "T=0.1 always returns max token");
        CHECK(hist[0] == 0 && hist[1] == 0 && hist[2] == 0 && hist[4] == 0,
              "T=0.1 never returns non-max tokens");
        tensor_free(logits);
    }

    /* 2b. T=1.0 抽样分布大致等于原始分布 */
    {
        float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        Tensor* logits = make_logits(5, data);
        int hist[5] = {0};
        srand(123);
        hist_temperature(logits, 1.0f, 5000, hist, 5);
        /* softmax 后大致是 [0.03, 0.06, 0.12, 0.24, 0.55]
         * 5000 次抽样期望：位置 4 ~2750，位置 0 ~150
         */
        CHECK(hist[4] > hist[3], "T=1.0: max logit sampled most");
        CHECK(hist[0] < hist[4], "T=1.0: min logit sampled least");
        CHECK(hist[0] + hist[1] + hist[2] + hist[3] + hist[4] == 5000,
              "T=1.0: total samples is 5000");
        tensor_free(logits);
    }

    /* 2c. T=2.0 让分布摊平（最大 logit 的概率显著下降） */
    {
        float data[] = {0.0f, 0.0f, 0.0f, 0.0f, 5.0f};
        Tensor* logits = make_logits(5, data);
        int hist[5] = {0};
        srand(7);
        hist_temperature(logits, 2.0f, 5000, hist, 5);
        /* T=2.0 后 scaled = [0, 0, 0, 0, 2.5]
         * softmax 后大致是 [0.07, 0.07, 0.07, 0.07, 0.72]
         * 位置 4 应仍然最多，但比 T=1.0 时少很多
         */
        CHECK(hist[4] < 4500, "T=2.0: max loses some mass");
        CHECK(hist[0] + hist[1] + hist[2] + hist[3] > 500,
              "T=2.0: non-max tokens get significant mass");
        tensor_free(logits);
    }

    /* 2d. T <= 0 退化为 greedy */
    {
        float data[] = {1.0f, 5.0f, 2.0f, 3.0f};
        Tensor* logits = make_logits(4, data);
        int idx = student_sample_temperature(logits, 0.0f);
        CHECK(idx == 1, "T=0 falls back to greedy");
        int idx2 = student_sample_temperature(logits, -1.0f);
        CHECK(idx2 == 1, "T<0 falls back to greedy");
        tensor_free(logits);
    }
}

/* ============== TEST 3: student_sample_top_k ============== */
static void test_student_sample_top_k(void) {
    printf("\n=== TEST 3: student_sample_top_k ===\n");

    /* 3a. K <= 0 退化为 temperature */
    {
        float data[] = {1.0f, 5.0f, 2.0f};
        Tensor* logits = make_logits(3, data);
        int idx = student_sample_top_k(logits, 0, 0.1f);
        /* T=0.1 时几乎必选位置 1 */
        CHECK(idx == 1, "K=0 falls back to temperature sample (T=0.1 → greedy)");
        tensor_free(logits);
    }

    /* 3b. K >= vocab_size 等于不限制 */
    {
        float data[] = {1.0f, 5.0f, 2.0f, 3.0f};
        Tensor* logits = make_logits(4, data);
        int hist[4] = {0};
        srand(99);
        /* K=10 >= 4，退化为 temperature sample（T=0.1 → greedy） */
        hist_top_k(logits, 10, 0.1f, 200, hist, 4);
        CHECK(hist[1] == 200, "K>=V: equivalent to no filter (always picks max)");
        tensor_free(logits);
    }

    /* 3c. K=2 只在前两个最高 logit 中抽样 */
    {
        float data[] = {1.0f, 5.0f, 2.0f, 4.0f, 0.0f};
        Tensor* logits = make_logits(5, data);
        int hist[5] = {0};
        srand(2024);
        /* 前 2 高 = 位置 1 (5.0) 和 位置 3 (4.0)
         * T=0.1 → 几乎都选位置 1（因为它比 4.0 高）
         */
        hist_top_k(logits, 2, 0.1f, 500, hist, 5);
        CHECK(hist[0] == 0 && hist[2] == 0 && hist[4] == 0,
              "K=2: positions outside top-2 never sampled");
        CHECK(hist[1] + hist[3] == 500, "K=2: only top-2 positions sampled");
        CHECK(hist[1] > hist[3], "K=2 + T=0.1: position 1 sampled more than position 3");
        tensor_free(logits);
    }

    /* 3d. 平局 top-K 的处理：前 2 都一样时，两个都应该被采到 */
    {
        float data[] = {3.0f, 3.0f, 3.0f, 0.0f};
        Tensor* logits = make_logits(4, data);
        int hist[4] = {0};
        srand(2025);
        /* K=2 + T=0.5：在前 2 高里抽样。前 2 高 = 位置 0、1（按 argmax first-wins 约定） */
        hist_top_k(logits, 2, 0.5f, 2000, hist, 4);
        CHECK(hist[2] == 0 && hist[3] == 0,
              "K=2 + tied top: positions outside top-2 never sampled");
        CHECK(hist[0] + hist[1] == 2000, "K=2 + tied top: only top-2 positions sampled");
        tensor_free(logits);
    }
}

/* ============== 入口 ============== */
int verify_run_all(void) {
    printf("========================================\n");
    printf("  Lab08 - 文本生成 测试\n");
    printf("========================================\n");

    test_student_sample_greedy();
    test_student_sample_temperature();
    test_student_sample_top_k();

    printf("\n========================================\n");
    printf("测试结果: %d 通过, %d 失败\n", g_pass, g_fail);
    printf("========================================\n");

    return (g_fail == 0) ? 0 : 1;
}
