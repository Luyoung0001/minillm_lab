#include "student.h"
#include "verify.h"

#include "tensor.h"
#include "math_ops.h"
#include "loss.h"
#include "optimizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* TODO(student): 实现序列平均交叉熵损失
 *
 * 公式（数值稳定版）:
 *   对每个位置 s:
 *     max_val = max_i(logits[s][i])
 *     log_sum_exp = max_val + log(sum_i(exp(logits[s][i] - max_val)))
 *     L_s = -logits[s][target_s] + log_sum_exp
 *   L = mean(L_s, s = 0..seq_len-1)
 *
 * 写 8~15 行
 */
float student_cross_entropy_loss(
    Tensor* logits,
    int* targets,
    int seq_len,
    int vocab_size
) {
    /* TODO(student): 外层循环 seq_len，内层循环 vocab_size 算 log_sum_exp，累加 L_s 后除以 seq_len */
    (void)logits; (void)targets; (void)seq_len; (void)vocab_size;
    return 0.0f;
}

/* TODO(student): 实现单位置 softmax + cross-entropy 梯度
 *
 * 三步走:
 *   1. 找 max_val = max(logits[i])
 *   2. 对每个 i 算 exp(logits[i] - max_val)，存到 grad[i]，累加 sum
 *   3. grad[i] = grad[i] / sum；若 i == target 则 grad[i] -= 1
 *
 * 提示：可以用一个临时 buffer 存 exp 值，也可以直接覆盖 grad
 * 写 6~12 行
 */
void student_softmax_ce_grad(
    float* logits,
    int target,
    int vocab_size,
    float* grad
) {
    /* TODO(student): 找 max → exp(x - max)/sum → 减 one_hot(target) */
    (void)logits; (void)target; (void)vocab_size; (void)grad;
}

/* TODO(student): 对单个张量应用 Adam 一步
 *
 * 六行公式（按顺序，对每个 i）:
 *   m[i]  = beta1 * m[i] + (1 - beta1) * grad[i]
 *   v[i]  = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i]
 *   bc1   = 1 - beta1^t            # 在循环外算
 *   bc2   = 1 - beta2^t
 *   m_hat = m[i] / bc1
 *   v_hat = v[i] / bc2
 *   param[i] -= lr * m_hat / (sqrtf(v_hat) + eps)
 *   param[i] -= lr * weight_decay * param[i]   # AdamW
 *
 * 写 8~12 行
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
) {
    /* TODO(student): 循环外算 bc1 / bc2；循环内依次更新 m / v / param */
    (void)param; (void)m; (void)v; (void)grad; (void)size;
    (void)lr; (void)beta1; (void)beta2; (void)eps;
    (void)weight_decay; (void)t;
    return 0;
}

int main(void) {
    return verify_run_all();
}
