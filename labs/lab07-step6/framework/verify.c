#include "verify.h"
#include "student.h"

#include "tensor.h"
#include "math_ops.h"
#include "loss.h"
#include "optimizer.h"

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

#define EPS 1e-3f

/* ============== helper：均匀 logits ============== */
/* 把 logits 填成全 0（softmax 后是均匀分布） */
static void fill_uniform_logits(Tensor* t) {
    for (int i = 0; i < t->size; i++) {
        t->data[i] = 0.0f;
    }
}

/* ============== helper：单位置 logits[vocab]，target 位置设为 5.0 ============== */
static void make_target_dominant(float* logits, int vocab_size, int target) {
    for (int i = 0; i < vocab_size; i++) {
        logits[i] = (i == target) ? 5.0f : 0.0f;
    }
}

/* ============== TEST 1: student_cross_entropy_loss ============== */
static void test_student_cross_entropy_loss(void) {
    printf("\n=== TEST 1: student_cross_entropy_loss ===\n");

    /* 1a. 均匀 logits → loss ≈ log(vocab_size) */
    int vocab_size = 10;
    int seq_len = 4;
    int shape[] = {seq_len, vocab_size};
    Tensor* logits = tensor_zeros(2, shape);
    fill_uniform_logits(logits);

    int targets[] = {3, 7, 1, 5};
    float loss = student_cross_entropy_loss(logits, targets, seq_len, vocab_size);

    CHECK(loss > 0.0f, "cross_entropy > 0");
    CHECK(fabsf(loss - logf((float)vocab_size)) < EPS,
          "loss ~ log(vocab_size) for uniform logits");

    /* 1b. target 位置 logit 主导 → loss 接近 0 */
    /* logits: 全 0，但 targets[s] 位置设成 5.0 */
    for (int s = 0; s < seq_len; s++) {
        for (int i = 0; i < vocab_size; i++) {
            logits->data[s * vocab_size + i] = 0.0f;
        }
        logits->data[s * vocab_size + targets[s]] = 5.0f;
    }
    float loss2 = student_cross_entropy_loss(logits, targets, seq_len, vocab_size);
    CHECK(loss2 < 0.5f, "loss small when target logit dominates");

    /* 1c. 跟 framework 参考实现做差 */
    float loss_ref = cross_entropy_loss(logits, targets, seq_len, vocab_size);
    CHECK(fabsf(loss2 - loss_ref) < 1e-2f, "matches framework cross_entropy_loss within 0.01");

    tensor_free(logits);
}

/* ============== TEST 2: student_softmax_ce_grad ============== */
static void test_student_softmax_ce_grad(void) {
    printf("\n=== TEST 2: student_softmax_ce_grad ===\n");

    int vocab_size = 6;
    int target = 2;
    float logits[] = {1.0f, 2.0f, 3.0f, 0.5f, 0.0f, 1.5f};
    float grad[6] = {0};
    float grad_ref[6] = {0};

    student_softmax_ce_grad(logits, target, vocab_size, grad);

    /* 跟 framework reference 对比（用 softmax_cross_entropy_grad） */
    softmax_cross_entropy_grad(logits, target, vocab_size, grad_ref);
    int close = 1;
    for (int i = 0; i < vocab_size; i++) {
        if (fabsf(grad[i] - grad_ref[i]) > 1e-4f) { close = 0; break; }
    }
    CHECK(close, "grad matches framework reference");

    /* grad 之和应该 ≈ 0（softmax 减去 one-hot 的求和恒为 0） */
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) sum += grad[i];
    CHECK(fabsf(sum) < 1e-4f, "grad sums to ~0");

    /* target 位置应该是负数（因为减了 1） */
    CHECK(grad[target] < 0.0f, "grad at target is negative");

    /* 非 target 位置应该都是正数 */
    int all_pos = 1;
    for (int i = 0; i < vocab_size; i++) {
        if (i == target) continue;
        if (grad[i] <= 0.0f) { all_pos = 0; break; }
    }
    CHECK(all_pos, "grad at non-target positions is positive");
}

/* ============== TEST 3: student_adam_step ============== */
static void test_student_adam_step(void) {
    printf("\n=== TEST 3: student_adam_step ===\n");

    int size = 8;
    float param[8];
    float m[8];
    float v[8];
    float grad[8];

    /* 构造一个简单场景：param 全 1，grad 全 0.1，期望 param 减小 */
    for (int i = 0; i < size; i++) {
        param[i] = 1.0f;
        m[i] = 0.0f;
        v[i] = 0.0f;
        grad[i] = 0.1f;
    }

    int rc = student_adam_step(
        param, m, v, grad, size,
        0.01f,   /* lr */
        0.9f,    /* beta1 */
        0.999f,  /* beta2 */
        1e-8f,   /* eps */
        0.0f,    /* weight_decay */
        1        /* t = 1 */
    );
    CHECK(rc == 0, "adam_step returns 0");

    /* 3a. m / v 已被更新 */
    int m_updated = 0, v_updated = 0;
    for (int i = 0; i < size; i++) {
        if (fabsf(m[i] - 0.01f) > 1e-5f && m[i] != 0.0f) m_updated++;
        if (fabsf(v[i] - 0.001f) > 1e-5f && v[i] != 0.0f) v_updated++;
    }
    CHECK(m_updated > 0, "m is updated after one step");
    CHECK(v_updated > 0, "v is updated after one step");

    /* 3b. param 被减小（grad 是正的，lr>0，所以 param 减） */
    int all_smaller = 1;
    for (int i = 0; i < size; i++) {
        if (param[i] >= 1.0f) { all_smaller = 0; break; }
    }
    CHECK(all_smaller, "param decreases when grad > 0");

    /* 3c. 多步：t=2 时 m、v 继续增大（因为 grad 一直是正的） */
    float param2[8], m2[8], v2[8], grad2[8];
    for (int i = 0; i < size; i++) {
        param2[i] = 1.0f;
        m2[i] = 0.0f;
        v2[i] = 0.0f;
        grad2[i] = 0.1f;
    }
    /* 手动算 1 步后用 t=2 再算 1 步 */
    student_adam_step(param2, m2, v2, grad2, size,
        0.01f, 0.9f, 0.999f, 1e-8f, 0.0f, 1);
    student_adam_step(param2, m2, v2, grad2, size,
        0.01f, 0.9f, 0.999f, 1e-8f, 0.0f, 2);
    int all_smaller2 = 1;
    for (int i = 0; i < size; i++) {
        if (param2[i] >= 1.0f) { all_smaller2 = 0; break; }
    }
    CHECK(all_smaller2, "param keeps decreasing at t=2");

    /* 3d. weight_decay 让 param 再额外减小（如果学员实现正确） */
    float param3[8], m3[8], v3[8], grad3[8];
    for (int i = 0; i < size; i++) {
        param3[i] = 1.0f;
        m3[i] = 0.0f;
        v3[i] = 0.0f;
        grad3[i] = 0.0f;   /* 零梯度，只看 weight_decay 的影响 */
    }
    student_adam_step(param3, m3, v3, grad3, size,
        0.1f, 0.9f, 0.999f, 1e-8f, 0.5f, 1);  /* weight_decay=0.5 */
    int all_smaller3 = 1;
    for (int i = 0; i < size; i++) {
        if (param3[i] >= 1.0f) { all_smaller3 = 0; break; }
    }
    CHECK(all_smaller3, "weight_decay shrinks param even when grad=0");
}

/* ============== 入口 ============== */
int verify_run_all(void) {
    printf("========================================\n");
    printf("  Lab07 - 训练循环 测试\n");
    printf("========================================\n");

    test_student_cross_entropy_loss();
    test_student_softmax_ce_grad();
    test_student_adam_step();

    printf("\n========================================\n");
    printf("测试结果: %d 通过, %d 失败\n", g_pass, g_fail);
    printf("========================================\n");

    return (g_fail == 0) ? 0 : 1;
}
