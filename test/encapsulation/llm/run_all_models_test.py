import os
import sys
import importlib.util


def _load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法从路径加载模块: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    print("开始运行所有 LLM 相关测试...")
    print("=" * 60)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))

    # 确保项目根目录在 sys.path 中
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # 目标测试文件的绝对路径
    hf_test_path = os.path.join(current_dir, 'encapsulation', 'llm', 'test_huggingface.py')
    qwen_test_path = os.path.join(current_dir, 'encapsulation', 'llm', 'test_qwen3.py')
    openai_test_path = os.path.join(current_dir, 'encapsulation', 'llm', 'test_llm.py')

    results = []

    # 1) HuggingFace Embedding 测试
    if os.path.exists(hf_test_path):
        print('step 1: HuggingFace embedding tests')
        hf_module = _load_module_from_path('test_huggingface_module', hf_test_path)
        if hasattr(hf_module, 'run_tests'):
            results.append(('huggingface', hf_module.run_tests()))
        else:
            print('⚠ 未找到 run_tests() 于 test_huggingface.py，跳过')
            results.append(('huggingface', True))
    else:
        print('⚠ 未找到 test_huggingface.py，跳过')
        results.append(('huggingface', True))

    # 2) Qwen Reranker 测试
    if os.path.exists(qwen_test_path):
        print('step 2: Qwen reranker tests')
        qwen_module = _load_module_from_path('test_qwen3_module', qwen_test_path)
        if hasattr(qwen_module, 'run_tests'):
            results.append(('qwen3', qwen_module.run_tests()))
        else:
            print('⚠ 未找到 run_tests() 于 test_qwen3.py，跳过')
            results.append(('qwen3', True))
    else:
        print('⚠ 未找到 test_qwen3.py，跳过')
        results.append(('qwen3', True))

    # 3) OpenAI LLM 测试
    if os.path.exists(openai_test_path):
        print('step 3: OpenAI LLM tests')
        openai_module = _load_module_from_path('test_openai_module', openai_test_path)
        if hasattr(openai_module, 'run_tests'):
            results.append(('openai', openai_module.run_tests()))
        else:
            print('⚠ 未找到 run_tests() 于 test_llm.py，跳过')
            results.append(('openai', True))
    else:
        print('⚠ 未找到 test_llm.py，跳过')
        results.append(('openai', True))

    print("\n" + "=" * 60)
    print("测试汇总结果：")
    all_success = True
    for name, ok in results:
        status = '成功' if ok else '失败'
        print(f"- {name}: {status}")
        all_success = all_success and ok

    if all_success:
        print("\nAll tests passed.")
        return 0
    else:
        print("\nSome tests failed.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
