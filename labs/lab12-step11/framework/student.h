#ifndef STUDENT_H
#define STUDENT_H

/*
 * BPE 训练循环里"找最高频相邻 pair"的最小实现。
 *
 * 学员从一段 int 数组（即 token id 序列）出发，遍历所有相邻对
 * (tokens[i], tokens[i+1])，把出现次数最多的那一对通过 out 参数返回，
 * 并把次数作为返回值。当 len < 2 或最高频次 < 2 时返回 0。
 */
int student_bpe_count_pairs(const int* tokens, int len,
                            int* out_first, int* out_second);

/*
 * 把序列里所有"值为 (first, second) 的相邻 pair"原地替换成 new_id。
 *
 * 返回一个新分配的 int 数组（caller 负责 free），新长度写进 *out_len。
 * 长度只会变小或不变（每个 pair 折叠成 1 个 token）。
 */
int* student_bpe_apply_merge(const int* tokens, int len,
                              int first, int second, int new_id,
                              int* out_len);

#endif // STUDENT_H
