// Chapter 06 迁移兼容头：旧代码中的 KVCache 实际只保存词元历史。
#pragma once

#include "chapter06_generation/token_history.h"

namespace chapter06_generation {

using KVCache = TokenHistory;

}  // namespace chapter06_generation
