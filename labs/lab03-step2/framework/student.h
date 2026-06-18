#ifndef STUDENT_H
#define STUDENT_H

#include "embedding.h"

/*
 * student_pe_sinusoidal - 计算单个位置的 sinusoidal PE 值
 *
 * 公式:
 *   PE(pos, 2i)   = sin(pos / 10000^(2i/d))
 *   PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
 *   其中 i = dim / 2,  d = hidden_dim
 *
 * 返回: PE(pos, dim) 这一个 float 值
 *
 * 提示: pos=0 时所有维度都是 0 或 1, 是检验公式最简单的 case
 */
float student_pe_sinusoidal(int pos, int dim, int hidden_dim);

/*
 * student_embedding_forward - 完整 embedding 前向传播
 *
 * @param emb       Embedding 句柄 (内含 token_embedding + position_embedding)
 * @param token_ids 输入 token ID 数组, 长度 = seq_len
 * @param seq_len   序列长度
 * @param output    预分配的输出张量, shape = [seq_len, hidden_dim]
 *                  本函数把 token_emb[id] + PE[pos] 写入 output[pos, :]
 *
 * 边界:
 *   - 任一指针为 NULL -> return
 *   - seq_len > emb->max_seq_len -> stderr 警告 + return
 *   - 越界 token id -> 用 1 (UNK) 替代
 */
void  student_embedding_forward(Embedding* emb, int* token_ids, int seq_len, Tensor* output);

#endif // STUDENT_H
