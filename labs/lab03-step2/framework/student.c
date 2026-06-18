/*
 * Lab03 - 词嵌入与位置编码
 *
 * 学员要做的全部代码都在本文件里. 框架 (../../framework/...) 走 Makefile 链接.
 *
 * 你要实现的 2 个函数 (见 student.h):
 *   1) student_pe_sinusoidal  - 单元素 PE
 *   2) student_embedding_forward - 完整 forward (查 token + 加 PE)
 *
 * 关键概念:
 *   - token_embedding 形状 [vocab_size, hidden_dim],  查表用
 *   - position_embedding 形状 [max_seq_len, hidden_dim],  按 pos 取一行
 *   - forward: out[pos, d] = token_emb[id, d] + PE(pos, d)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "student.h"
#include "verify.h"
#include "embedding.h"     /* 共享的 Embedding 类型 */

/* ============================================================
 * TODO(student): 实现 student_pe_sinusoidal
 *
 * 签名: float student_pe_sinusoidal(int pos, int dim, int hidden_dim);
 *
 * 行为 (一行公式):
 *   i     = dim / 2
 *   angle = pos / 10000^(2i / hidden_dim)
 *   若 dim 是偶数 -> 返回 sinf(angle)
 *   若 dim 是奇数 -> 返回 cosf(angle)
 *
 * 检查点 (手算):
 *   student_pe_sinusoidal(0, 0, 64) = 0.0   // sin(0)
 *   student_pe_sinusoidal(0, 1, 64) = 1.0   // cos(0)
 *   student_pe_sinusoidal(0, 2, 64) = 0.0   // sin(0)
 *   student_pe_sinusoidal(0, 3, 64) = 1.0   // cos(0)
 *
 * 提示: 写 5~10 行. 需要 #include <math.h> 里的 sinf/cosf/powf.
 * ============================================================ */
float student_pe_sinusoidal(int pos, int dim, int hidden_dim) {
    /* TODO(student): replace this stub with your 5~10 line implementation */
    (void)pos;
    (void)dim;
    (void)hidden_dim;
    return 0.0f;          /* 占位: 永远返回 0, 让 verify TEST 1/2 失败提醒你来改 */
}

/* ============================================================
 * TODO(student): 实现 student_embedding_forward
 *
 * 签名: void student_embedding_forward(Embedding* emb, int* token_ids,
 *                                       int seq_len, Tensor* output);
 *
 * 行为:
 *   1) NULL 保护: emb / token_ids / output 任一为 NULL -> 直接 return
 *   2) 越界保护: seq_len > emb->max_seq_len -> fprintf(stderr, ...) + return
 *   3) hidden_dim = emb->hidden_dim
 *   4) for pos in [0, seq_len):
 *        id = token_ids[pos]
 *        if (id < 0 || id >= emb->vocab_size) id = 1   // UNK
 *        for d in [0, hidden_dim):
 *           tok = emb->token_embedding->data[id * hidden_dim + d]
 *           pe  = student_pe_sinusoidal(pos, d, hidden_dim)
 *           output->data[pos * hidden_dim + d] = tok + pe
 *
 * 提示: 写 8~14 行. flat 索引 = (row) * hidden_dim + d
 * ============================================================ */
void student_embedding_forward(Embedding* emb, int* token_ids, int seq_len, Tensor* output) {
    /* TODO(student): replace this stub with your 8~14 line implementation */
    (void)emb;
    (void)token_ids;
    (void)seq_len;
    (void)output;
    /* 占位: 什么都不写, 让 verify TEST 3/4 失败提醒你来改 */
}

/* ============================================================
 * main - 把控制权交给 verify
 * 学员不要改 main. 想加临时 printf 调试可以临时加, 提交前删掉.
 * ============================================================ */
int main(void) {
    return verify_run_all();
}
