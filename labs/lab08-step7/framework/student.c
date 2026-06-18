#include "student.h"
#include "verify.h"

#include "tensor.h"
#include "math_ops.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* TODO(student): 实现 greedy = argmax
 *
 * 思路：
 *   1. 遍历 logits->size 个位置，记 best_val 和 best_idx
 *   2. 遇到更大的 logits[i] 就更新
 *   3. 平局时取先出现的（>= 而不是 >）
 *
 * 也可以直接调 framework 的 tensor_argmax(logits)——
 * 但自己写一遍更利于理解"采样 = 概率论里最简单的决策"
 *
 * 写 5~10 行
 */
int student_sample_greedy(Tensor* logits) {
    /* TODO(student): 遍历找 argmax */
    (void)logits;
    return 0;
}

/* ============== helper：softmax 后按累积分布抽样 ============== */
/* 在 probs[0..vocab_size) 上做累积分布抽样，返回抽样得到的索引
 * 学员函数可以复用这个 helper（如果实现了它）
 * 也可自己写一份内联在 student_sample_temperature 里
 */
int student_sample_from_probs(Tensor* probs) {
    int vocab_size = probs->size;
    if (vocab_size <= 0) return 0;

    /* 计算累积和 */
    float cum = 0.0f;
    float* cdf = (float*)malloc((size_t)vocab_size * sizeof(float));
    if (cdf == NULL) return 0;

    for (int i = 0; i < vocab_size; i++) {
        cum += probs->data[i];
        cdf[i] = cum;
    }

    /* 处理浮点误差：最后一个 cdf 可能略小于 1.0，归一到 1 */
    if (vocab_size > 0 && cdf[vocab_size - 1] > 0.0f) {
        float total = cdf[vocab_size - 1];
        for (int i = 0; i < vocab_size; i++) {
            cdf[i] /= total;
        }
    }

    /* 抽 u ~ Uniform(0, 1)，找第一个 cdf[i] >= u 的 i */
    float u = (float)rand() / (float)RAND_MAX;
    int chosen = vocab_size - 1;  /* 兜底：u 接近 1 时返回最后一个 */
    for (int i = 0; i < vocab_size; i++) {
        if (cdf[i] >= u) {
            chosen = i;
            break;
        }
    }

    free(cdf);
    return chosen;
}

/* TODO(student): 实现温度采样
 *
 * 数学（两步）：
 *   1. scaled[i] = logits[i] / temperature
 *   2. probs = softmax(scaled)
 *   3. 在 probs 上按累积分布抽样
 *
 * 边界：
 *   - temperature <= 0：退化为 student_sample_greedy(logits)
 *   - 用 framework 提供的 softmax(probs, scaled)
 *   - 自己实现累积抽样（参考上面的 student_sample_from_probs）
 *
 * 写 12~18 行
 */
int student_sample_temperature(Tensor* logits, float temperature) {
    /* TODO(student): 温度 <= 0 走 greedy；否则 scaled = logits/T → softmax → 累积抽样 */
    (void)logits;
    (void)temperature;
    return 0;
}

/* TODO(student): 实现 Top-K 采样
 *
 * 思路：
 *   1. 复制 logits 到 filtered
 *   2. 把不属于 top-K 的位置置为 -INFINITY
 *   3. 在 filtered 上做"温度采样"
 *
 * 边界：
 *   - k <= 0 或 k >= vocab_size：退化为 student_sample_temperature(logits, temperature)
 *
 * 实现提示 1（找门槛）：
 *   复制 logits 到 tmp，循环 k 次找 argmax，记阈值 threshold = 选中的最小值
 *
 * 实现提示 2（直接排序）：
 *   把 logits 排序，前 K 个保留、其余置 -INF（需要 stdlib 的 qsort + 一个 index 数组）
 *
 * 写 12~20 行
 */
int student_sample_top_k(Tensor* logits, int k, float temperature) {
    /* TODO(student): k 边界 → temperature；否则 filtered = logits 但非 top-K 置 -INF → 温度采样 */
    (void)logits;
    (void)k;
    (void)temperature;
    return 0;
}

int main(void) {
    return verify_run_all();
}
