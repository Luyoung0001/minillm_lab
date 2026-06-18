#ifndef STUDENT_H
#define STUDENT_H

#include "model.h"
#include "config.h"

/* 返回默认的 ModelConfig，6 字段（vocab/hidden/heads/layers/ffn/max_seq） */
ModelConfig student_default_config(void);

/* 申请一个完整 GPTModel：embedding + N x block + final LN + lm_head */
GPTModel* student_model_create(ModelConfig config);

/* 把整个 model 写盘：magic "MLLM" + version + config + 全部权重 */
int student_model_save(GPTModel* model, const char* path);

#endif // STUDENT_H
