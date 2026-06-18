#include "student.h"
#include "verify.h"

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

/* helper：把一个 Tensor 的全部 float 写出去 */
static void save_tensor(FILE* f, Tensor* t) {
    if (t == NULL || t->data == NULL) return;
    fwrite(t->data, sizeof(float), (size_t)t->size, f);
}

/* ============================================================
 * 任务 1.5.1：填 6 个字段
 * ============================================================ */
ModelConfig student_default_config(void) {
    ModelConfig config = {0};
    // TODO(student): 填 6 个 config 字段
    //   vocab_size / hidden_dim / num_heads / num_layers
    //   ffn_dim / max_seq_len
    //   提醒：hidden_dim 必须能整除 num_heads（config_validate 会检查）
    return config;
}

/* ============================================================
 * 任务 1.5.2：申请 embedding + N 层 block + final LN + lm_head
 * ============================================================ */
GPTModel* student_model_create(ModelConfig config) {
    // TODO(student): 申请 GPTModel* 并装配 5 个子对象
    //   1) malloc(sizeof(GPTModel))，保存 config
    //   2) embedding_create(vocab_size, hidden_dim, max_seq_len)
    //   3) 循环 num_layers 次 transformer_block_create(...)
    //   4) layernorm_create(hidden_dim, 1e-5f)
    //   5) tensor_zeros(2, {hidden_dim, vocab_size})  -- 注意 shape 顺序
    //   任一步失败要 free 前面已申请的内存，再 return NULL
    return NULL;
}

/* ============================================================
 * 任务 1.5.3：写 magic + version + config + 全部权重
 * ============================================================ */
int student_model_save(GPTModel* model, const char* path) {
    // TODO(student): 按下面顺序写
    //   [4B magic = 0x4D4C4C4D ("MLLM")]
    //   [4B version = 1]
    //   [sizeof(ModelConfig) 字节]
    //   [token_embedding][position_embedding]
    //   [for each layer: LN1(γ,β) + Attn(Wq,Wk,Wv,Wo) + LN2(γ,β) + FFN(W1,b1,W2,b2)]
    //   [final LN(γ,β)]
    //   [lm_head]
    //   失败 return -1；成功 return 0
    return -1;
}

/* ============================================================
 * 学员 main
 * ============================================================ */
int main(void) {
    return verify_run_all();
}
