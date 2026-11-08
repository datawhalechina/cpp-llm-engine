import { defineConfig } from 'vitepress'

export default defineConfig({
  title: '从零构建 C++ LLM 推理引擎',
  description: '从 GGUF、Tensor、Decoder 到 KV Cache、调度与推理服务',
  lang: 'zh-CN',
  cleanUrls: false,
  srcExclude: ['maintenance/**/*.md'],
  lastUpdated: true,
  ignoreDeadLinks: [
    /^\.\.\/_code\//,
    /^\.\/\.\.\/_code\//,
  ],
  vite: {
    publicDir: '.vitepress/public',
    resolve: {
      preserveSymlinks: true,
    },
  },
  locales: {
    '/': {
      lang: 'zh-CN',
      label: '中文',
      title: '从零构建 C++ LLM 推理引擎',
      description: '从 GGUF、Tensor、Decoder 到 KV Cache、调度与推理服务',
    },
    '/en/': {
      lang: 'en-US',
      label: 'English',
      title: 'C++ LLM Inference Engine Tutorial',
      description: 'Build an LLM inference engine from scratch',
    },
  },
  themeConfig: {
    sidebar: createSidebar(),
    nav: [
      { text: '首页', link: '/' },
      { text: '教程主入口', link: '/README' },
      { text: '英文入口', link: '/en/README' },
      { text: '使用文档', link: '/guide' },
      { text: 'Foundation 题库', link: '/exercises/foundation/README' },
      { text: '开发参考', link: '/guides/README' },
      { text: '设计参考', link: '/reference/README' },
      { text: '代码索引', link: '/code/README' },
    ],
    locales: {
      '/en/': {
        nav: [
          { text: 'Home', link: '/en/' },
          { text: 'Tutorial Entry', link: '/en/README' },
          { text: 'Guide', link: '/en/guide' },
          { text: 'Development', link: '/en/guides/README' },
          { text: 'Reference', link: '/en/reference/README' },
        ],
        sidebar: createEnglishSidebar(),
      },
    },
  },
})

type SidebarItem = {
  text: string
  link?: string
  collapsed?: boolean
  items?: SidebarItem[]
}

function numberedText(index: string | number, text: string) {
  return `${index} ${text}`
}

function chapterGroup(chapterCode: string, text: string, base: string, pages: Array<{ text: string; file: string; section?: boolean }>) {
  return {
    text,
    collapsed: true,
    items: [
      { text: numberedText(`${chapterCode}.0`, '本章导读'), link: `${base}/intro` },
      ...pages.map((page) => {
        const pageNumber = page.file.match(/^(\d+)_/)?.[1]
        return {
          text: page.section || !pageNumber
            ? page.text
            : numberedText(`${chapterCode}.${Number.parseInt(pageNumber, 10)}`, page.text),
          link: `${base}/${page.file.replace(/\.md$/, '')}`,
        }
      }),
    ],
  }
}

function codeGroup(chapterCode: string, text: string, base: string) {
  return {
    text,
    collapsed: true,
    items: [
      { text: `${chapterCode} 代码索引`, link: `${base}` },
    ],
  }
}

function chapterLandingGroup(chapterCode: string, text: string, base: string) {
  return {
    text,
    collapsed: true,
    items: [
      { text: numberedText(`${chapterCode}.0`, 'Intro'), link: `/en${base}/intro` },
      { text: numberedText(`${chapterCode}.1`, 'Summary'), link: `/en${base}/summary` },
    ],
  }
}

function createEnglishSidebar() {
  return {
    '/en/': [
      {
        text: 'Tutorial Guide',
        collapsed: false,
        items: [
          { text: 'Home', link: '/en/' },
          { text: 'Tutorial Entry', link: '/en/README' },
          { text: 'Guide', link: '/en/guide' },
          { text: 'Development References', link: '/en/guides/README' },
          { text: 'Design References', link: '/en/reference/README' },
          { text: 'Foundation', link: '/en/foundation/intro' },
          chapterLandingGroup('00', 'Chapter 00 Quickstart', '/chapter00_quickstart'),
          chapterLandingGroup('01', 'Chapter 01 Model Weights and Tensor Objects', '/chapter01_tensor_loader'),
          chapterLandingGroup('02', 'Chapter 02 Tensor Views and Shared Storage', '/chapter02_tensor_ops'),
          chapterLandingGroup('03', 'Chapter 03 MatMul and Compute Backends', '/chapter03_matmul'),
          chapterLandingGroup('04', 'Chapter 04 Decoder Structure and Attention', '/chapter04_attention'),
          chapterLandingGroup('05', 'Chapter 05 Quantization', '/chapter05_quantization'),
          chapterLandingGroup('06', 'Chapter 06 Generation Runtime and Serving Basics', '/chapter06_generation'),
        ],
      },
    ],
  }
}

function createSidebar() {
  return {
    '/': [
      {
        text: '开始这里',
        collapsed: false,
        items: [
          { text: '主入口', link: '/README' },
          { text: '使用文档', link: '/guide' },
          { text: 'CPU 六任务学习计划', link: '/guides/phase1_cpu_learning_plan' },
          { text: 'Foundation 题库', link: '/exercises/foundation/README' },
          chapterGroup('00', '第 00 章 起跑线', '/chapter00_quickstart', [
            { text: '环境准备', file: '01_environment_setup.md' },
            { text: '克隆并编译 llama.cpp', file: '02_clone_and_build_llamacpp.md' },
            { text: '下载 TinyLlama', file: '03_download_tinyllama.md' },
            { text: '第一次推理', file: '04_first_inference.md' },
            { text: '排查问题', file: '05_troubleshooting.md' },
          ]),
          {
            text: '基础准备',
            collapsed: true,
            items: [
              { text: numberedText(1, '导读'), link: '/foundation/intro' },
              { text: numberedText(1.5, '正文与练习映射'), link: '/foundation/exercise_map' },
              { text: numberedText(2, 'C++ 基础'), link: '/foundation/01_cxx_basics' },
              { text: numberedText(3, 'CMake 和编译基础'), link: '/foundation/02_cmake_build_basics' },
              { text: numberedText(4, 'CUDA 和环境基础'), link: '/foundation/03_cuda_environment_basics' },
              { text: numberedText(5, '调试与测试基础'), link: '/foundation/04_debugging_testing_basics' },
              { text: numberedText(6, 'OOP 和设计基础'), link: '/foundation/05_oop_and_design_basics' },
              { text: numberedText(7, '数据结构和算法基础'), link: '/foundation/06_data_structure_algorithm_basics' },
              { text: numberedText(8, 'CUDA 常用库'), link: '/foundation/07_cuda_common_libraries' },
              { text: numberedText(9, 'PyTorch 到 C++ 桥接'), link: '/foundation/08_pytorch_to_cpp_bridge' },
              { text: numberedText(10, '最小 C++ / CUDA 项目'), link: '/foundation/09_minimal_cpp_cuda_project' },
              { text: numberedText(11, '进程、线程与同步异步'), link: '/foundation/10_concurrency_basics' },
            ],
          },
        ],
      },
      {
        text: '引擎主线',
        collapsed: false,
        items: [
          chapterGroup('01', '第 01 章 模型权重与 Tensor 对象', '/chapter01_tensor_loader', [
            { text: 'Tensor 对象', file: '01_tensor_object.md' },
            { text: '选读：教学权重格式', file: '02_teaching_weight_format.md' },
            { text: '模型加载与验证', file: '03_model_loader_validation.md' },
            { text: '输入契约与 Tokenizer', file: '04_input_contract_and_tokenizer.md' },
            { text: '模型元数据与 GGUF', file: '05_model_metadata_and_gguf.md' },
            { text: '进阶：Loader 框架对照', file: '06_framework_loader_comparison.md' },
          ]),
          chapterGroup('02', '第 02 章 Tensor 视图与共享存储', '/chapter02_tensor_ops', [
            { text: '共享存储', file: '01_shared_ptr_memory.md' },
            { text: 'reshape', file: '02_reshape.md' },
            { text: 'slice', file: '03_slice.md' },
            { text: '单元测试', file: '04_unit_tests.md' },
            { text: 'Tensor 与 View 框架对照', file: '05_framework_tensor_comparison.md' },
          ]),
          chapterGroup('03', '第 03 章 MatMul 与计算后端', '/chapter03_matmul', [
            { text: '算子基类', file: '01_operator_base.md' },
            { text: 'MatMul 设计', file: '02_matmul_design.md' },
            { text: 'CPU 参考实现', file: '03_cpu_reference.md' },
            { text: 'cuBLAS 实现', file: '04_cublas_implement.md' },
            { text: '选读：用 PyTorch 独立复算', file: '05_test_with_pytorch.md' },
            { text: '复杂度直觉', file: '06_complexity_intuition.md' },
            { text: '从 MatMul 到真实推理 backend', file: '07_runtime_backend_bridge.md' },
          ]),
          chapterGroup('04', '第 04 章 Decoder 结构与 Attention', '/chapter04_attention', [
            { text: '多头注意力', file: '01_multihead_attention.md' },
            { text: 'LayerNorm 和激活', file: '02_layernorm_and_activation.md' },
            { text: '前馈网络', file: '03_feedforward.md' },
            { text: 'DecoderLayer', file: '04_decoder_layer.md' },
            { text: '加载第一层权重', file: '05_load_first_layer.md' },
            { text: '真实 TinyLlama Decoder', file: '06_real_tinyllama_decoder.md' },
            { text: 'Decoder 成熟框架对照', file: '07_framework_comparison.md' },
          ]),
          chapterGroup('05', '第 05 章 量化', '/chapter05_quantization', [
            { text: '量化基础', file: '01_quantization_basics.md' },
            { text: '表达量化数据', file: '02_extend_tensor.md' },
            { text: '接收量化权重', file: '03_load_quantized_weights.md' },
            { text: '反量化算子', file: '04_dequantize_op.md' },
            { text: '量化线性层', file: '05_linear_with_quant.md' },
            { text: '精度验证', file: '06_accuracy_validation.md' },
            { text: '量化成熟框架对照', file: '07_tensorrt_gemma_vllm_comparison.md' },
            { text: '量化在生成运行时中的位置', file: '08_quantization_in_serving_runtime.md' },
          ]),
          chapterGroup('06', '第 06 章 Generation Core', '/chapter06_generation/next', [
            { text: '请求与执行契约', file: '01_request_contract.md' },
            { text: 'Prefill 与 Decode', file: '02_prefill_decode.md' },
            { text: '单请求生成循环', file: '03_single_request_loop.md' },
            { text: '请求生命周期', file: '04_request_lifecycle.md' },
            { text: '采样与停止', file: '05_sampling_stop.md' },
            { text: '单请求 K/V 语义', file: '06_single_request_kv_semantics.md' },
            { text: '流式输出与错误', file: '07_streaming_errors.md' },
            { text: '真实 TinyLlama 生成', file: '08_real_tinyllama_generation.md' },
            { text: '框架对照', file: '09_framework_comparison.md' },
          ]),
        ],
      },
      {
        text: '验证与优化',
        collapsed: false,
        items: [
          chapterGroup('07', '第 07 章 K/V Cache 与内存管理', '/chapter07_kv_cache/next', [
            { text: 'K/V 形状与连续缓存', file: '01_kv_baseline.md' },
            { text: '块表与物理内存', file: '02_block_mapping.md' },
            { text: '分页 K/V 事务', file: '03_paged_transaction.md' },
            { text: '分页 Attention', file: '04_paged_attention.md' },
            { text: '真实模型迁移', file: '05_real_model_migration.md' },
            { text: '生产扩展', file: '06_production_extensions.md' },
          ]),
          chapterGroup('08', '第 08 章 调度与批处理', '/chapter08_scheduling/next', [
            { text: '调度基线', file: '01_scheduler_baseline.md' },
            { text: 'Step Batching', file: '02_step_batching.md' },
            { text: 'Chunked Prefill 与优先级', file: '03_chunked_priority.md' },
            { text: '真实 Tensor Batch', file: '04_tensor_batch.md' },
            { text: 'Overlap Scheduling', file: '05_overlap_scheduling.md' },
            { text: '框架对照', file: '06_framework_comparison.md' },
          ]),
          chapterGroup('09', '第 09 章 性能证据与优化', '/chapter09_performance/next', [
            { text: '正确性与工作负载', file: '01_correctness_workload.md' },
            { text: 'Benchmark 入口', file: '02_benchmark_setup.md' },
            { text: '性能指标', file: '03_metrics.md' },
            { text: '统计与记录', file: '04_statistics_schema.md' },
            { text: 'Profiling 工具', file: '05_profiling_tools.md' },
            { text: '热点归因', file: '06_hotspot_attribution.md' },
            { text: 'A/B 实验', file: '07_ab_experiment.md' },
            { text: '决策与回归', file: '08_decision_regression.md' },
            { text: '框架对照', file: '09_framework_comparison.md' },
          ]),
          chapterGroup('10', '第 10 章 推理服务', '/chapter10_serving/next', [
            { text: 'ServiceCore', file: '01_service_core.md' },
            { text: '准入与背压', file: '02_admission_backpressure.md' },
            { text: '终态释放', file: '03_terminal_release.md' },
            { text: 'Transport', file: '04_transport.md' },
            { text: 'Completion', file: '05_completion_request.md' },
            { text: 'SSE 与断连', file: '06_sse_disconnect.md' },
            { text: '并发 Worker', file: '07_concurrency_workers.md' },
            { text: '可观测性', file: '08_observability.md' },
            { text: '真实模型验证', file: '09_real_model_validation.md' },
            { text: '框架对照', file: '10_framework_comparison.md' },
          ]),
        ],
      },
      {
        text: '查阅与扩展',
        collapsed: true,
        items: [
          { text: '开发参考总入口', link: '/guides/README' },
          { text: 'CPU 六任务学习计划', link: '/guides/phase1_cpu_learning_plan' },
          { text: 'CMake 常用技巧', link: '/guides/cmake_guide' },
          { text: 'CUDA 安装与配置', link: '/guides/setup_cuda' },
          { text: 'GGUF 解析', link: '/guides/gguf_parser' },
          { text: 'Foundation 题库', link: '/exercises/foundation/README' },
          { text: '设计参考总入口', link: '/reference/README' },
          { text: '章节文档与代码资产', link: '/reference/chapter_assets' },
          { text: '实验与验证资产', link: '/reference/experiment_assets' },
          { text: '训练框架 vs 推理框架', link: '/reference/framework_comparison' },
          { text: '数据结构与算法资产表', link: '/reference/data_structures_algorithms' },
          { text: '推理引擎内存管理', link: '/reference/memory_management' },
          { text: 'CPU / GPU 主线与契约', link: '/reference/cpu_gpu_architecture' },
          { text: '多模态学习指南', link: '/guides/multimodal_inference' },
          { text: '模型基线与扩展顺序', link: '/reference/model_support' },
          { text: '公式索引', link: '/reference/formulas' },
          { text: '术语表', link: '/reference/glossary' },
          { text: '继续学习', link: '/reference/ecosystem/README#完成主线后怎样继续' },
          { text: 'FAQ', link: '/reference/faq' },
          { text: '生态对照', link: '/reference/ecosystem/README' },
        ],
      },
    ],
    '/code/': [
      {
        text: '代码索引',
        collapsed: false,
        items: [
          { text: '总览', link: '/code/README' },
          { text: 'Repository Engine First Success', link: '/code/engine' },
          { text: 'CUDA 公共基础层', link: '/code/cuda' },
          { text: '多模态 CPU 输入扩展', link: '/code/multimodal' },
          codeGroup('00', '第 00 章 起跑线', '/code/chapter00_quickstart'),
          codeGroup('01', '第 01 章 模型权重与 Tensor 对象', '/code/chapter01_tensor_loader'),
          codeGroup('02', '第 02 章 Tensor 视图与共享存储', '/code/chapter02_tensor_ops'),
          codeGroup('03', '第 03 章 MatMul 与计算后端', '/code/chapter03_matmul'),
          codeGroup('04', '第 04 章 Decoder 结构与 Attention', '/code/chapter04_attention'),
          codeGroup('05', '第 05 章 量化', '/code/chapter05_quantization'),
          codeGroup('06', '第 06 章 Generation Core', '/code/chapter06_generation'),
          codeGroup('07', '第 07 章 K/V Cache', '/code/chapter07_kv_cache'),
          codeGroup('08', '第 08 章 调度与批处理', '/code/chapter08_scheduling'),
          codeGroup('09', '第 09 章 性能证据', '/code/chapter09_performance'),
          codeGroup('10', '第 10 章 推理服务', '/code/chapter10_serving'),
        ],
      },
    ],
  }
}
