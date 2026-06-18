#include "student.h"
#include "verify.h"

#include "tensor.h"
#include "math_ops.h"
#include "layernorm.h"
#include "ffn.h"
#include "attention.h"
#include "transformer.h"

#include <math.h>
#include <stdio.h>

/* TODO(student): 实现 1D LayerNorm
 * 公式: y_i = (x_i - mean) / sqrt(var + eps)
 * x 是 [hidden_dim] 的一维张量；y 已预分配，与 x 同 shape
 * 不乘 gamma、不加 beta（基础版）
 * 写 5~15 行
 */
int student_layernorm(Tensor* x, Tensor* y, float eps) {
    /* TODO(student): 计算 x 在最后一维的均值与方差，逐元素归一化到 y */
    (void)x; (void)y; (void)eps;
    return 0;
}

/* TODO(student): 实现原地残差加法 a += b
 * a 与 b 同 shape
 * 写 2~5 行（提示：tensor_add_inplace）
 */
int student_residual_add(Tensor* a, Tensor* b) {
    /* TODO(student): 调用 tensor_add_inplace(a, b) */
    (void)a; (void)b;
    return 0;
}

/* TODO(student): 实现 Pre-LN Transformer Block
 * 数据流：
 *   h = input
 *   h = h + Attn(LN(h))     // 第一次残差
 *   h = h + FFN(LN(h))      // 第二次残差
 *   output = h
 *
 * block_ptr 实际类型是 TransformerBlock*
 * cache_ptr 实际类型是 TransformerCache*
 * 注意用 (TransformerBlock*) / (TransformerCache*) 强转后访问成员
 * 写 10~20 行
 */
int student_block_forward(
    void* block_ptr,
    Tensor* input,
    Tensor* mask,
    void* cache_ptr,
    Tensor* output
) {
    /* TODO(student): 拼 LN + Attn + 残差 + LN + FFN + 残差 */
    (void)block_ptr; (void)input; (void)mask; (void)cache_ptr; (void)output;
    return 0;
}

int main(void) {
    return verify_run_all();
}
