#ifndef STUDENT_H
#define STUDENT_H

#include "tensor.h"

/*
 * student_tensor_get - 用 stride 公式计算一维偏移并返回值
 *
 * 给定 2D 张量 t（shape=[rows, cols], strides=[cols, 1]）和坐标 indices={i, j}，
 * 一维偏移 = indices[0] * strides[0] + indices[1] * strides[1]
 *
 * @return t->data[offset]; 越界时返回 0
 */
float student_tensor_get(Tensor* t, int* indices);

/*
 * student_tensor_set - 把 value 写入张量 t 在 indices 位置
 *
 * 用与 student_tensor_get 对称的 stride 公式
 */
void  student_tensor_set(Tensor* t, int* indices, float value);

/*
 * student_softmax_stable - 数值稳定 softmax
 *
 * softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
 *
 * @param in   输入数组
 * @param out  输出数组（与 in 同长度）
 * @param n    数组长度
 *
 * 提示: 先算 max，再写 exp，再归一化。3 步。
 */
void  student_softmax_stable(const float* in, float* out, int n);

#endif // STUDENT_H
