#include "student.h"
#include "verify.h"
#include "tensor.h"
#include "math_ops.h"
#include "attention.h"
#include <math.h>

/*
 * 单头 attention scores: out[i][j] = scale * sum_d Q[i][d] * K[j][d]
 * 输入 Q, K 形状: [seq_len, head_dim]
 * 输出 out 形状: [seq_len, seq_len] (需预先分配)
 *
 * TODO(student): 用三层循环 (i, j, d) 计算 Q @ K^T，再乘以 scale。
 *                不要把 K 转置——按 K->data[j * head_dim + d] 读即可。
 *                5~15 行。
 */
int student_attention_scores(Tensor* Q, Tensor* K, float scale, Tensor* out) {
    if (Q == NULL || K == NULL || out == NULL) return -1;
    if (Q->ndim != 2 || K->ndim != 2 || out->ndim != 2) return -1;
    if (Q->shape[1] != K->shape[1]) return -1;       // head_dim 必须一致
    if (Q->shape[0] != out->shape[0]) return -1;     // seq_len 一致
    if (K->shape[0] != out->shape[1]) return -1;

    // TODO(student): 在这里写实现
    (void)scale;
    return 0;
}

/*
 * 原地加 mask: scores[i][j] += mask[i][j]
 * 两个张量形状必须一致。
 *
 * TODO(student): 1 个 for 循环搞定。3~5 行。
 */
void student_apply_mask(Tensor* scores, Tensor* mask) {
    if (scores == NULL || mask == NULL) return;
    if (scores->size != mask->size) return;

    // TODO(student): 在这里写实现
}

/*
 * 沿最后一维 softmax: out[row][j] = exp(in[row][j] - max) / sum exp(in[row][k] - max)
 *
 * TODO(student):
 *   1) 用 in->shape[in->ndim - 1] 拿到最后一维长度 last_dim
 *   2) 外层循环每一行 (rows = in->size / last_dim)
 *   3) 每行: 找 max -> exp(x - max) -> 求和 -> 归一化
 *   10~20 行。
 */
void student_softmax(Tensor* out, Tensor* in) {
    if (out == NULL || in == NULL) return;
    if (in->size != out->size) return;
    if (in->ndim < 1) return;

    // TODO(student): 在这里写实现
}

int main(void) {
    return verify_run_all();
}