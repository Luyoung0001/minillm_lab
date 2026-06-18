/*
 * Lab01 verify - 自动验证 3 个 student 函数
 *
 * 约束: 本文件只用 framework 的公开 API (tensor_create_2d / tensor_free /
 * tensor_get / tensor_set) + student 暴露的 3 个函数. 不读 Tensor 内部字段.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "student.h"
#include "tensor.h"
#include "verify.h"

static int g_failures = 0;

#define TEST_BEGIN(n) do { printf("[TEST %d] ", (n)); fflush(stdout); } while (0)
#define TEST_PASS()    do { printf(" ... [PASS]\n"); } while (0)
#define TEST_FAIL(fmt, ...) do {                                           \
    printf(" ... [FAIL] " fmt "\n", ##__VA_ARGS__); g_failures++;           \
} while (0)

/* ---------- TEST 1: student_tensor_get ---------- */
static void test_tensor_get(void) {
    TEST_BEGIN(1);
    Tensor* t = tensor_create_2d(2, 3);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++) {
            int idx[2] = {i, j};
            tensor_set(t, idx, (float)(i * 10 + j));
        }

    /* expected: [[0,1,2],[10,11,12]] */
    int idx0[2] = {0, 0}, idx1[2] = {1, 2};
    float v00 = student_tensor_get(t, idx0);
    float v12 = student_tensor_get(t, idx1);
    if (fabsf(v00 - 0.0f) > 1e-5f || fabsf(v12 - 12.0f) > 1e-5f) {
        TEST_FAIL("get(0,0)=%f (want 0), get(1,2)=%f (want 12)", v00, v12);
    } else {
        TEST_PASS();
    }
    tensor_free(t);
}

/* ---------- TEST 2: student_tensor_set ---------- */
static void test_tensor_set(void) {
    TEST_BEGIN(2);
    Tensor* t = tensor_create_2d(2, 3);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++) {
            int idx[2] = {i, j};
            student_tensor_set(t, idx, 7.0f);
        }

    /* 全部应为 7.0f */
    int ok = 1;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++) {
            int idx[2] = {i, j};
            if (fabsf(tensor_get(t, idx) - 7.0f) > 1e-5f) ok = 0;
        }

    if (!ok) TEST_FAIL("set did not write correctly");
    else      TEST_PASS();
    tensor_free(t);
}

/* ---------- TEST 3: student_softmax_stable ---------- */
static void test_softmax(void) {
    TEST_BEGIN(3);
    float in[]   = {1.0f, 2.0f, 3.0f};
    float out[3] = {0};
    student_softmax_stable(in, out, 3);

    /* 手算: exp(0)/Z + exp(1)/Z + exp(2)/Z = 1 */
    float sum = out[0] + out[1] + out[2];
    if (fabsf(sum - 1.0f) > 1e-4f) {
        TEST_FAIL("sum(out) = %f (want 1.0)", sum);
    } else if (out[2] <= out[1] || out[1] <= out[0]) {
        TEST_FAIL("monotonicity broken: %f %f %f", out[0], out[1], out[2]);
    } else {
        TEST_PASS();
    }
}

/* ---------- TEST 4: softmax 对大数不爆 ---------- */
static void test_softmax_large(void) {
    TEST_BEGIN(4);
    float in[]   = {1000.0f, 1001.0f, 1002.0f};
    float out[3] = {0};
    student_softmax_stable(in, out, 3);

    int has_nan = 0;
    for (int i = 0; i < 3; i++) if (isnan(out[i]) || isinf(out[i])) has_nan = 1;
    if (has_nan) {
        TEST_FAIL("softmax overflowed (NaN/Inf detected)");
    } else {
        TEST_PASS();
    }
}

int verify_run_all(void) {
    g_failures = 0;
    printf("==========================================\n");
    printf("       Lab01 - 张量与数学运算 验证       \n");
    printf("==========================================\n");
    test_tensor_get();
    test_tensor_set();
    test_softmax();
    test_softmax_large();
    printf("==========================================\n");
    if (g_failures == 0) {
        printf("All tests passed!\n");
        return 0;
    } else {
        printf("%d test(s) FAILED.\n", g_failures);
        return 1;
    }
}
