#include "verify.h"
#include "student.h"

#include "tensor.h"
#include "math_ops.h"
#include "attention.h"

#include <stdio.h>
#include <math.h>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg)                                                      \
    do {                                                                      \
        if (cond) {                                                           \
            printf("  [PASS] %s\n", msg);                                     \
            g_pass++;                                                         \
        } else {                                                              \
            printf("  [FAIL] %s\n", msg);                                     \
            g_fail++;                                                         \
        }                                                                     \
    } while (0)

#define FLOAT_EQ(a, b) (fabsf((a) - (b)) < 1e-4f)

/* ===== TEST 1: student_attention_scores ===== */
static void test_attention_scores(void) {
    printf("\n[TEST 1] student_attention_scores 基础形状与缩放 ...\n");

    /* Q = [[1, 0], [0, 1]], K = [[1, 0], [0, 1]], scale = 0.5
     * Q @ K^T = [[1, 0], [0, 1]]
     * * 0.5   = [[0.5, 0], [0, 0.5]]
     */
    int seq_len = 2, head_dim = 2;
    int q_shape[] = {seq_len, head_dim};
    float q_data[] = {1.0f, 0.0f, 0.0f, 1.0f};
    Tensor* Q = tensor_create(2, q_shape);
    tensor_fill_data(Q, q_data);

    Tensor* K = tensor_create(2, q_shape);
    tensor_fill_data(K, q_data);

    int out_shape[] = {seq_len, seq_len};
    Tensor* out = tensor_zeros(2, out_shape);

    int rc = student_attention_scores(Q, K, 0.5f, out);
    CHECK(rc == 0, "返回 0 (成功)");

    int zero_idx[] = {0, 0};
    CHECK(tensor_shape_equal(out, out) && out->shape[0] == 2 && out->shape[1] == 2,
          "输出形状为 [2, 2]");

    int i00[] = {0, 0};
    int i01[] = {0, 1};
    int i10[] = {1, 0};
    int i11[] = {1, 1};
    CHECK(FLOAT_EQ(tensor_get(out, i00), 0.5f), "out[0][0] = 0.5");
    CHECK(FLOAT_EQ(tensor_get(out, i01), 0.0f), "out[0][1] = 0");
    CHECK(FLOAT_EQ(tensor_get(out, i10), 0.0f), "out[1][0] = 0");
    CHECK(FLOAT_EQ(tensor_get(out, i11), 0.5f), "out[1][1] = 0.5");
    (void)zero_idx;

    /* 第二个小用例: seq_len=3, head_dim=3, scale=1.0
     * Q = I, K = I -> out = I (主对角 1, 其他 0)
     */
    int s2 = 3, h2 = 3;
    int sh2[] = {s2, h2};
    float eye3[] = {1, 0, 0,
                    0, 1, 0,
                    0, 0, 1};
    Tensor* Q2 = tensor_create(2, sh2);
    Tensor* K2 = tensor_create(2, sh2);
    tensor_fill_data(Q2, eye3);
    tensor_fill_data(K2, eye3);
    int out2_shape[] = {s2, s2};
    Tensor* out2 = tensor_zeros(2, out2_shape);
    rc = student_attention_scores(Q2, K2, 1.0f, out2);
    CHECK(rc == 0, "seq_len=3 调用成功");
    int a00[] = {0, 0};
    int a01[] = {0, 1};
    CHECK(FLOAT_EQ(tensor_get(out2, a00), 1.0f), "I @ I 在 (0,0) = 1");
    CHECK(FLOAT_EQ(tensor_get(out2, a01), 0.0f), "I @ I 在 (0,1) = 0");

    tensor_free(Q);
    tensor_free(K);
    tensor_free(out);
    tensor_free(Q2);
    tensor_free(K2);
    tensor_free(out2);
}

/* ===== TEST 2: student_apply_mask ===== */
static void test_apply_mask(void) {
    printf("\n[TEST 2] student_apply_mask 因果掩码 ...\n");

    /* seq_len = 3 因果 mask:
     * [ 0, -1e9, -1e9 ]
     * [ 0,    0, -1e9 ]
     * [ 0,    0,    0 ]
     */
    Tensor* scores = tensor_zeros(2, (int[]){3, 3});
    Tensor* mask = create_causal_mask(3);

    student_apply_mask(scores, mask);

    int i00[] = {0, 0};
    int i01[] = {0, 1};
    int i02[] = {0, 2};
    int i11[] = {1, 1};
    int i12[] = {1, 2};
    int i22[] = {2, 2};

    CHECK(FLOAT_EQ(tensor_get(scores, i00), 0.0f),       "mask 后 [0][0] = 0");
    CHECK(tensor_get(scores, i01) < -1e8f,               "mask 后 [0][1] = -1e9");
    CHECK(tensor_get(scores, i02) < -1e8f,               "mask 后 [0][2] = -1e9");
    CHECK(FLOAT_EQ(tensor_get(scores, i11), 0.0f),       "mask 后 [1][1] = 0");
    CHECK(tensor_get(scores, i12) < -1e8f,               "mask 后 [1][2] = -1e9");
    CHECK(FLOAT_EQ(tensor_get(scores, i22), 0.0f),       "mask 后 [2][2] = 0");

    tensor_free(scores);
    tensor_free(mask);
}

/* ===== TEST 3: student_softmax ===== */
static void test_softmax(void) {
    printf("\n[TEST 3] student_softmax 行和 = 1 ...\n");

    /* 全 0 输入 -> softmax = 1/N  */
    int s = 4;
    Tensor* in0 = tensor_zeros(2, (int[]){s, s});
    Tensor* out0 = tensor_zeros(2, (int[]){s, s});
    student_softmax(out0, in0);

    int row_sum_ok = 1;
    for (int i = 0; i < s; i++) {
        float sum = 0.0f;
        for (int j = 0; j < s; j++) {
            int idx[] = {i, j};
            sum += tensor_get(out0, idx);
        }
        if (fabsf(sum - 1.0f) > 1e-4f) {
            row_sum_ok = 0;
            printf("    row %d sum = %.6f (期望 1.0)\n", i, sum);
        }
    }
    CHECK(row_sum_ok, "全 0 输入 softmax 后每行和 ≈ 1.0");

    /* softmax([1, 1, 1]) = [1/3, 1/3, 1/3]  */
    Tensor* in1 = tensor_create(2, (int[]){1, 3});
    tensor_fill_data(in1, (float[]){1.0f, 1.0f, 1.0f});
    Tensor* out1 = tensor_zeros(2, (int[]){1, 3});
    student_softmax(out1, in1);
    int i0[] = {0, 0};
    int i1[] = {0, 1};
    int i2[] = {0, 2};
    CHECK(FLOAT_EQ(tensor_get(out1, i0), 1.0f / 3.0f), "softmax([1,1,1])[0] = 1/3");
    CHECK(FLOAT_EQ(tensor_get(out1, i1), 1.0f / 3.0f), "softmax([1,1,1])[1] = 1/3");
    CHECK(FLOAT_EQ(tensor_get(out1, i2), 1.0f / 3.0f), "softmax([1,1,1])[2] = 1/3");

    tensor_free(in0);
    tensor_free(out0);
    tensor_free(in1);
    tensor_free(out1);
}

/* ===== TEST 4: 端到端 scores -> mask -> softmax ===== */
static void test_end_to_end(void) {
    printf("\n[TEST 4] 端到端: scores → mask → softmax ...\n");

    /* 用 Q=I, K=I, scale=0.5 -> scores = 0.5 * I
     * 加因果 mask 后下三角 = 0.5, 上三角 = -1e9
     * softmax 后每行只有一个非零 (对角) -> 1, 上三角 = 0
     */
    int seq_len = 4, head_dim = 4;
    float eye4[] = {1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1};
    Tensor* Q = tensor_create(2, (int[]){seq_len, head_dim});
    Tensor* K = tensor_create(2, (int[]){seq_len, head_dim});
    tensor_fill_data(Q, eye4);
    tensor_fill_data(K, eye4);

    Tensor* scores = tensor_zeros(2, (int[]){seq_len, seq_len});
    int rc = student_attention_scores(Q, K, 0.5f, scores);
    CHECK(rc == 0, "scores 计算成功");

    Tensor* mask = create_causal_mask(seq_len);
    student_apply_mask(scores, mask);

    Tensor* attn = tensor_zeros(2, (int[]){seq_len, seq_len});
    student_softmax(attn, scores);

    /* 行和 = 1 */
    int row_sum_ok = 1;
    for (int i = 0; i < seq_len; i++) {
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            int idx[] = {i, j};
            sum += tensor_get(attn, idx);
        }
        if (fabsf(sum - 1.0f) > 1e-4f) {
            row_sum_ok = 0;
            printf("    row %d sum = %.6f\n", i, sum);
        }
    }
    CHECK(row_sum_ok, "attn 行和 = 1");

    /* 上三角 = 0 (j > i) */
    int upper_zero = 1;
    for (int i = 0; i < seq_len; i++) {
        for (int j = i + 1; j < seq_len; j++) {
            int idx[] = {i, j};
            if (tensor_get(attn, idx) > 1e-6f) {
                upper_zero = 0;
            }
        }
    }
    CHECK(upper_zero, "attn 上三角 = 0（mask 生效）");

    tensor_free(Q);
    tensor_free(K);
    tensor_free(scores);
    tensor_free(mask);
    tensor_free(attn);
}

int verify_run_all(void) {
    printf("========================================\n");
    printf("   Lab04 多头自注意力 — verify\n");
    printf("========================================\n");

    g_pass = 0;
    g_fail = 0;

    test_attention_scores();
    test_apply_mask();
    test_softmax();
    test_end_to_end();

    printf("\n========================================\n");
    if (g_fail == 0) {
        printf("结果: 全 PASS  (%d 个断言)\n", g_pass);
    } else {
        printf("结果: %d 失败 / %d 通过\n", g_fail, g_pass);
    }
    printf("========================================\n");

    return g_fail == 0 ? 0 : 1;
}