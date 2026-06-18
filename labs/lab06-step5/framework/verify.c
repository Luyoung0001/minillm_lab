#include "verify.h"
#include "student.h"

#include "model.h"
#include "config.h"
#include "embedding.h"
#include "transformer.h"
#include "layernorm.h"
#include "tensor.h"
#include "math_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; printf("  [PASS] %s\n", msg); } \
    else      { g_fail++; printf("  [FAIL] %s\n", msg); } \
} while (0)

/* ------------------------------------------------------------------
 * TEST 1: student_default_config returns valid 6-field config
 * ------------------------------------------------------------------ */
static void test_default_config(void) {
    printf("[TEST 1] student_default_config returns a valid 6-field config\n");
    ModelConfig c = student_default_config();
    CHECK(c.vocab_size   > 0,  "vocab_size > 0");
    CHECK(c.hidden_dim   > 0,  "hidden_dim > 0");
    CHECK(c.num_heads    > 0,  "num_heads > 0");
    CHECK(c.num_layers   > 0,  "num_layers > 0");
    CHECK(c.ffn_dim      > 0,  "ffn_dim > 0");
    CHECK(c.max_seq_len  > 0,  "max_seq_len > 0");
    CHECK(c.num_heads > 0 && c.hidden_dim % c.num_heads == 0,
          "hidden_dim divisible by num_heads");
}

/* ------------------------------------------------------------------
 * TEST 2: student_model_create wires all 5 sub-objects
 * ------------------------------------------------------------------ */
static void test_model_create(void) {
    printf("[TEST 2] student_model_create wires all 5 sub-objects\n");
    ModelConfig c = student_default_config();
    GPTModel* m = student_model_create(c);
    CHECK(m != NULL,                       "model_create returns non-NULL");
    CHECK(m != NULL && m->embedding != NULL, "embedding is created");
    CHECK(m != NULL && m->layers != NULL,  "layers array is allocated");
    CHECK(m != NULL && m->final_ln != NULL, "final LayerNorm is created");
    CHECK(m != NULL && m->lm_head != NULL, "lm_head tensor is created");
    if (m != NULL && m->lm_head != NULL) {
        int expected_size = c.vocab_size * c.hidden_dim;
        CHECK(m->lm_head->size == expected_size,
              "lm_head size = vocab_size * hidden_dim");
        CHECK(m->lm_head->ndim == 2, "lm_head is 2D");
        CHECK(m->lm_head->shape[0] == c.hidden_dim, "lm_head shape[0] = hidden_dim");
        CHECK(m->lm_head->shape[1] == c.vocab_size, "lm_head shape[1] = vocab_size");
    }
    if (m != NULL) {
        /* 检查每一层都被建出来 */
        int all_layers_ok = 1;
        for (int i = 0; i < c.num_layers; i++) {
            if (m->layers[i] == NULL) { all_layers_ok = 0; break; }
        }
        CHECK(all_layers_ok, "all num_layers transformer blocks are created");
    }
    if (m != NULL) model_free(m);
}

/* ------------------------------------------------------------------
 * TEST 3: model_save writes a non-trivial file (>240K floats)
 * ------------------------------------------------------------------ */
static void test_model_save_writes_file(void) {
    printf("[TEST 3] student_model_save writes a non-trivial file\n");
    ModelConfig c = student_default_config();
    GPTModel* m = student_model_create(c);
    if (m == NULL) { g_fail += 2; printf("  [FAIL] model_create returned NULL\n"); return; }
    /* 顺手做一下随机初始化，让权重的文件大小远大于 header */
    model_init_random(m, 0.02f);
    int rc = student_model_save(m, "build/test_model.bin");
    CHECK(rc == 0, "model_save returns 0 (success)");
    FILE* f = fopen("build/test_model.bin", "rb");
    CHECK(f != NULL, "file build/test_model.bin exists on disk");
    if (f != NULL) {
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fclose(f);
        /* header = 4 + 4 + 24 = 32 bytes; weights 240512 floats = 962048 bytes; 至少 > 940000 */
        CHECK(sz > 940000, "file size > 940KB (model weights present)");
    }
    model_free(m);
}

/* ------------------------------------------------------------------
 * TEST 4: file header is "MLLM" v1
 * ------------------------------------------------------------------ */
static void test_file_header(void) {
    printf("[TEST 4] model.bin header is \"MLLM\" v1\n");
    FILE* f = fopen("build/test_model.bin", "rb");
    CHECK(f != NULL, "can open saved file for header check");
    if (f != NULL) {
        uint32_t magic = 0, ver = 0;
        fread(&magic, sizeof(uint32_t), 1, f);
        fread(&ver,   sizeof(uint32_t), 1, f);
        fclose(f);
        CHECK(magic == 0x4D4C4C4D, "magic == 0x4D4C4C4D (\"MLLM\")");
        CHECK(ver == 1,             "version == 1");
    }
}

/* ------------------------------------------------------------------
 * TEST 5: save+load round-trip is bit-exact
 * ------------------------------------------------------------------ */
static void test_roundtrip_bit_exact(void) {
    printf("[TEST 5] save+load round-trip is bit-exact\n");
    ModelConfig c = student_default_config();
    GPTModel* m = student_model_create(c);
    CHECK(m != NULL, "model_create ok for round-trip");
    if (m == NULL) return;
    model_init_random(m, 0.02f);

    /* 用一段确定的 token 序列跑一次前向 */
    int ids[] = {1, 2, 3, 4, 5};
    int seq_len = 5;
    int log_shape[] = {seq_len, c.vocab_size};
    Tensor* logits_before = tensor_zeros(2, log_shape);
    model_forward(m, ids, seq_len, NULL, logits_before);

    /* 保存 */
    int rc = student_model_save(m, "build/test_roundtrip.bin");
    CHECK(rc == 0, "save succeeds");

    /* 销毁原模型 */
    model_free(m);

    /* 重新加载 */
    GPTModel* m2 = model_load("build/test_roundtrip.bin");
    CHECK(m2 != NULL, "load succeeds");
    if (m2 == NULL) { tensor_free(logits_before); return; }

    /* 再跑一次前向 */
    Tensor* logits_after = tensor_zeros(2, log_shape);
    model_forward(m2, ids, seq_len, NULL, logits_after);

    /* 逐位严格相等 */
    int diff = 0;
    for (int i = 0; i < logits_before->size; i++) {
        float a = tensor_get_flat(logits_before, i);
        float b = tensor_get_flat(logits_after,  i);
        if (a != b) diff++;
    }
    CHECK(diff == 0, "logits_before == logits_after (bit-exact)");

    tensor_free(logits_before);
    tensor_free(logits_after);
    model_free(m2);
}

int verify_run_all(void) {
    g_pass = 0; g_fail = 0;
    printf("========================================\n");
    printf("      Lab06 - 完整 GPT 模型 测试\n");
    printf("========================================\n");
    test_default_config();
    test_model_create();
    test_model_save_writes_file();
    test_file_header();
    test_roundtrip_bit_exact();
    printf("========================================\n");
    printf("测试结果: %d 通过, %d 失败\n", g_pass, g_fail);
    printf("========================================\n");
    return (g_fail == 0) ? 0 : 1;
}
