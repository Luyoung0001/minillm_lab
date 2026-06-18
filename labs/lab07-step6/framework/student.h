#ifndef STUDENT_H
#define STUDENT_H

#include "tensor.h"

/* 计算序列平均交叉熵损失
 * logits 形状 [seq_len, vocab_size]，targets 形状 [seq_len]
 * 返回标量 loss（float）
 */
float student_cross_entropy_loss(
    Tensor* logits,
    int* targets,
    int seq_len,
    int vocab_size
);

/* 对单个位置计算 softmax - one_hot 梯度
 * logits / grad 形状 [vocab_size]，target 是一个 int
 * 写完后 grad[i] = softmax(logits)[i] - (i == target ? 1 : 0)
 */
void student_softmax_ce_grad(
    float* logits,
    int target,
    int vocab_size,
    float* grad
);

/* 对单个一维参数张量应用一次 Adam 更新
 * param / m / v / grad 都是长度 size 的一维数组
 * t 是当前时间步（>= 1），从 1 开始
 * 返回 0 成功
 */
int student_adam_step(
    float* param,
    float* m,
    float* v,
    const float* grad,
    int size,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int t
);

#endif // STUDENT_H
