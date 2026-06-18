/*
 * Lab01 - 张量与数学运算
 *
 * 你要实现的 3 个函数 (见 student.h):
 *   1) student_tensor_get   - stride 公式
 *   2) student_tensor_set   - stride 公式（反向）
 *   3) student_softmax_stable - 数值稳定 softmax
 *
 * 框架 (../../framework/tensor.c, math_ops.c) 走 Makefile 链接.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "student.h"
#include "verify.h"

/* ============================================================
 * TODO(student): 实现 student_tensor_get
 *
 * 一维偏移 = indices[0] * t->strides[0] + indices[1] * t->strides[1]
 * 返回 t->data[offset]; 越界返回 0
 *
 * 提示: 写 3~5 行. 假设 2D tensor（ndim == 2）
 * ============================================================ */
float student_tensor_get(Tensor* t, int* indices) {
    /* TODO(student): replace this stub */
    (void)t; (void)indices;
    return 0.0f;   /* 占位 */
}

/* ============================================================
 * TODO(student): 实现 student_tensor_set
 * ============================================================ */
void student_tensor_set(Tensor* t, int* indices, float value) {
    /* TODO(student): replace this stub */
    (void)t; (void)indices; (void)value;
}

/* ============================================================
 * TODO(student): 实现 student_softmax_stable
 *
 * 步骤:
 *   1) 遍历 in 找 max
 *   2) 写 out[i] = expf(in[i] - max)
 *   3) sum 归一化: out[i] /= sum
 * ============================================================ */
void student_softmax_stable(const float* in, float* out, int n) {
    /* TODO(student): replace this stub */
    (void)in; (void)out; (void)n;
}

/* main 学员不要改 */
int main(void) {
    return verify_run_all();
}
