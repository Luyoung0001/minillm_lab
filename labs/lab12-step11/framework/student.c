/*
 * Lab12 - BPE 分词 (step11)
 *
 * 学员要做的：实现 BPE 训练循环里两个最关键的算子。
 *   1) student_bpe_count_pairs   —— 找最高频相邻 pair
 *   2) student_bpe_apply_merge   —— 在序列里替换 pair
 *
 * 跑法： make clean && make test
 */

#include "student.h"
#include "verify.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 * TODO 函数 1: 找最高频相邻 pair
 * ============================================================ */
int student_bpe_count_pairs(const int* tokens, int len,
                            int* out_first, int* out_second) {
    /* TODO(student):
     *   扫一遍相邻对 (tokens[i], tokens[i+1])，记录出现次数最多的那一对。
     *   - 频次最高的那一对写进 *out_first / *out_second；
     *   - 函数返回该 pair 的出现次数；
     *   - 边界：len < 2 或最高频 < 2 时返回 0，且不写 out 参数。
     *
     * 提示：O(n^2) 暴力即可；输入 < 几万个 token。
     */
    (void)tokens;
    (void)len;
    (void)out_first;
    (void)out_second;
    return 0;
}

/* ============================================================
 * TODO 函数 2: 在序列里替换 pair
 * ============================================================ */
int* student_bpe_apply_merge(const int* tokens, int len,
                              int first, int second, int new_id,
                              int* out_len) {
    /* TODO(student):
     *   新开一个 int 数组（长度 <= len），扫一遍：
     *     - 遇到 (tokens[i], tokens[i+1]) == (first, second)：
     *         写入 new_id，i++ 跳过下一个；
     *     - 否则照抄 tokens[i]。
     *   新长度写进 *out_len，返回新数组指针（caller free）。
     *
     * 提示：i < len-1 时才检查 pair；最后落单的那个 token 永远照抄。
     */
    (void)tokens;
    (void)len;
    (void)first;
    (void)second;
    (void)new_id;
    *out_len = 0;
    int* empty = (int*)malloc(sizeof(int));
    return empty;
}

/* ============================================================
 * main —— 学员无需改
 * ============================================================ */
int main(void) {
    return verify_run_all();
}
