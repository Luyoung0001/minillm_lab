#ifndef STUDENT_H
#define STUDENT_H

#include "tensor.h"

/*
 * 单头 attention scores: out[i][j] = scale * sum_d Q[i][d] * K[j][d]
 * 输入 Q, K 形状: [seq_len, head_dim]
 * 输出 out 形状: [seq_len, seq_len] (需预先分配)
 * 返回 0 表示成功，非 0 表示失败。
 */
int student_attention_scores(Tensor* Q, Tensor* K, float scale, Tensor* out);

/*
 * 原地加 mask: scores[i][j] += mask[i][j]
 * 两个张量形状必须一致; mask 通常是因果掩码 (下三角 0, 上三角 -1e9)
 */
void student_apply_mask(Tensor* scores, Tensor* mask);

/*
 * 沿最后一维 softmax: out[row][j] = exp(in[row][j] - max) / sum exp(in[row][k] - max)
 * 输入/输出张量形状相同 (任意 ndim, 但只对最后一维做 softmax)
 */
void student_softmax(Tensor* out, Tensor* in);

#endif // STUDENT_H