#ifndef STUDENT_H
#define STUDENT_H

#include "tensor.h"

/* 贪婪采样：返回 logits 张量 [vocab_size] 中最大值的索引
 * logits 不会被修改
 * 写 5~10 行
 */
int student_sample_greedy(Tensor* logits);

/* 温度采样：先把 logits 除以 temperature，再 softmax 后按概率抽样
 * temperature <= 0 时退化为 student_sample_greedy
 * logits 不会被修改；写 12~18 行
 */
int student_sample_temperature(Tensor* logits, float temperature);

/* Top-K 采样：只在前 K 个最大 logit 中按温度采样
 * k <= 0 或 k >= vocab_size 时退化为温度采样
 * logits 不会被修改；写 12~20 行
 */
int student_sample_top_k(Tensor* logits, int k, float temperature);

#endif // STUDENT_H
