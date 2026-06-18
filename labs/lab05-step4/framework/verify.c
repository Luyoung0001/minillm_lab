#include "verify.h"
#include "student.h"

#include "tensor.h"
#include "math_ops.h"
#include "layernorm.h"
#include "ffn.h"
#include "attention.h"
#include "transformer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* ============== 测试计数 ============== */
static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do {                                              \
    if (cond) {                                                            \
        printf("[PASS] %s\n", (msg));                                       \
        g_pass++;                                                          \
    } else {                                                               \
        printf("[FAIL] %s (line %d)\n", (msg), __LINE__);                  \
        g_fail++;                                                          \
    }                                                                      \
} while (0)

#define EPS 1e-4f

/* ============== TEST 1: student_layernorm ============== */
/* 测试 1D 向量归一化后均值≈0、方差≈1 */
static void test_student_layernorm(void) {
    printf("\n=== TEST 1: student_layernorm ===\n");
    int shape[] = {64};
    Tensor* x = tensor_randn(1, shape, 5.0f, 2.0f);
    Tensor* y = tensor_zeros(1, shape);

    int rc = student_layernorm(x, y, EPS);
    CHECK(rc == 0, "student_layernorm 返回 0");

    float mean = tensor_mean(y);
    float var  = tensor_var(y);
    CHECK(fabsf(mean) < 1e-3f, "1D LayerNorm 输出均值接近 0");
    CHECK(fabsf(var - 1.0f) < 1e-3f, "1D LayerNorm 输出方差接近 1");

    tensor_free(x);
    tensor_free(y);
}

/* ============== TEST 2: student_residual_add ============== */
/* 测试 a += b：先把 a 清零、b 置 1，结果应全 1 */
static void test_student_residual_add(void) {
    printf("\n=== TEST 2: student_residual_add ===\n");
    int shape[] = {8, 16};
    Tensor* a = tensor_zeros(2, shape);
    Tensor* b = tensor_ones(2, shape);

    int rc = student_residual_add(a, b);
    CHECK(rc == 0, "student_residual_add 返回 0");

    float sum = tensor_sum(a);
    CHECK(fabsf(sum - 128.0f) < 1e-3f, "残差加法 a += ones 之后 a 全为 1（sum=128）");

    /* 再加一次，应为 2 */
    student_residual_add(a, b);
    float sum2 = tensor_sum(a);
    CHECK(fabsf(sum2 - 256.0f) < 1e-3f, "残差加法第二次加之后 a 全为 2（sum=256）");

    tensor_free(a);
    tensor_free(b);
}

/* ============== TEST 3: student_block_forward ============== */
/* 测试 Pre-LN Block 前向：shape 对、不全零、与输入不同、无 NaN */
static void test_student_block_forward(void) {
    printf("\n=== TEST 3: student_block_forward ===\n");

    int hidden_dim = 64;
    int num_heads  = 4;
    int ffn_dim    = 256;
    int seq_len    = 8;

    /* 构造一个 TransformerBlock 与 cache */
    TransformerBlock* block = transformer_block_create(hidden_dim, num_heads, ffn_dim);
    transformer_block_init(block, 0.02f);
    TransformerCache* cache = transformer_cache_create(seq_len, hidden_dim, num_heads, ffn_dim);

    int shape[] = {seq_len, hidden_dim};
    Tensor* input  = tensor_randn(2, shape, 0.0f, 1.0f);
    Tensor* output = tensor_zeros(2, shape);
    Tensor* mask   = create_causal_mask(seq_len);

    int rc = student_block_forward((void*)block, input, mask, (void*)cache, output);
    CHECK(rc == 0, "student_block_forward 返回 0");

    /* shape 校验：output 与 input 同 shape */
    CHECK(output->ndim == input->ndim && output->size == input->size,
          "output shape 与 input shape 一致");

    /* 不全为零 */
    float out_sum = tensor_sum(output);
    CHECK(fabsf(out_sum) > 1e-6f, "output 不全为零");

    /* output 与 input 不同：两者差的 norm 大于 0 */
    Tensor* diff = tensor_zeros(2, shape);
    tensor_sub(diff, output, input);
    float diff_sum = tensor_sum(diff);
    CHECK(fabsf(diff_sum) > 1e-6f, "output 与 input 不同");

    /* 无 NaN：检查每个元素都是有限数 */
    int has_nan = 0;
    for (int i = 0; i < output->size; i++) {
        float v = tensor_get_flat(output, i);
        if (isnan(v) || isinf(v)) { has_nan = 1; break; }
    }
    CHECK(!has_nan, "output 不含 NaN / Inf");

    tensor_free(diff);
    tensor_free(input);
    tensor_free(output);
    tensor_free(mask);
    transformer_cache_free(cache);
    transformer_block_free(block);
}

/* ============== 入口 ============== */
int verify_run_all(void) {
    printf("========================================\n");
    printf("  Lab05 - Transformer Block 测试\n");
    printf("========================================\n");

    test_student_layernorm();
    test_student_residual_add();
    test_student_block_forward();

    printf("\n========================================\n");
    printf("测试结果: %d 通过, %d 失败\n", g_pass, g_fail);
    printf("========================================\n");

    return (g_fail == 0) ? 0 : 1;
}
