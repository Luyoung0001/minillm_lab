CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99 -I../../framework
LDFLAGS = -lm

SRC_DIR = framework
BUILD_DIR = build

OBJS = $(BUILD_DIR)/student.o $(BUILD_DIR)/verify.o
FRAMEWORK_OBJS = $(BUILD_DIR)/tensor.o $(BUILD_DIR)/math_ops.o $(BUILD_DIR)/tokenizer.o $(BUILD_DIR)/embedding.o $(BUILD_DIR)/attention.o $(BUILD_DIR)/layernorm.o $(BUILD_DIR)/ffn.o $(BUILD_DIR)/transformer.o $(BUILD_DIR)/model.o $(BUILD_DIR)/generate.o $(BUILD_DIR)/chat.o $(BUILD_DIR)/http_server.o

TARGET = student

.PHONY: all test clean
all: $(BUILD_DIR) $(TARGET)
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)
$(TARGET): $(OBJS) $(FRAMEWORK_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
$(BUILD_DIR)/student.o: $(SRC_DIR)/student.c $(SRC_DIR)/student.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@
$(BUILD_DIR)/verify.o: $(SRC_DIR)/verify.c $(SRC_DIR)/verify.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@
$(BUILD_DIR)/%.o: ../../framework/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@
test: $(TARGET)
	./$(TARGET)
clean:
	rm -rf $(BUILD_DIR) $(TARGET)
