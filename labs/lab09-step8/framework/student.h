#ifndef STUDENT_H
#define STUDENT_H

/*
 * 本 lab 不加载模型——只验证"两个字符串函数"。
 * 模型/tokenizer/真实 chat() 在 Lab10+ 才会用到。
 */

/* 把 (role, content) 数组按顺序拼成一段带 <|user|>/<|assistant|>/<|system|> 标签的 prompt。
 * 末尾追加一行 <|assistant|>\n（让模型知道在哪里开始接龙）。
 * 调用方负责 free 返回串。 */
char* student_format_prompt(const char** roles, const char** contents, int count);

/* 从模型原始输出里截出"最后一个 <|assistant|> 之后"的内容，
 * 在遇到下一个 <|user|>/<|assistant|>/<|system|>/<|end|> 时截断，并 trim 两端空白。
 * 调用方负责 free 返回串。 */
char* student_extract_response(const char* output);

#endif // STUDENT_H
