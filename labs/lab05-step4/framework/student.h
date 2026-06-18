#ifndef STUDENT_H
#define STUDENT_H

#include "tensor.h"

/* 学员要实现的函数：1D LayerNorm
 * 公式: y_i = (x_i - mean) / sqrt(var + eps)
 * 输入 x 是 1D 张量 [hidden_dim]，输出 y 与 x 同 shape
 * 不乘 gamma / 不加 beta（基础版）
 */
int student_layernorm(Tensor* x, Tensor* y, float eps);

/* 学员要实现的函数：原地残差加法
 * 原地修改 a: a += b；a 与 b shape 一致
 */
int student_residual_add(Tensor* a, Tensor* b);

/* 学员要实现的函数：Pre-LN Transformer Block
 * 数据流: h = h + Attn(LN(h)); h = h + FFN(LN(h)); output = h
 * 使用传入的 TransformerBlock 与 TransformerCache
 * output 由调用者预先分配
 */
int student_block_forward(
    void* block_ptr,        /* 实际类型 TransformerBlock* */
    Tensor* input,
    Tensor* mask,
    void* cache_ptr,        /* 实际类型 TransformerCache* */
    Tensor* output
);

#endif // STUDENT_H
