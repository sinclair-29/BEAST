import pickle as pkl
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("../LLMJailbreak/models/Mistral-7B-Instruct-v0.3")

# 加载 PKL 文件
file_path = 'data/mistral_k1=15_k2=15_length=40_0_10_ngram=1.pkl'
with open(file_path, 'rb') as f:
    log_data = pkl.load(f)

# --- 示例：分析第一个 prompt (索引为0) 的结果 ---
if 0 in log_data:
    result_for_prompt_0 = log_data[0]

    # 解包数据
    attack_results, _, time_spent = result_for_prompt_0
    curr_toks, curr_scores, best_toks, best_scores = attack_results

    print(f"对第一个 prompt 的攻击耗时: {time_spent:.2f} 秒")

    # 找到历史最佳的对抗性提示和它的分数
    # best_scores[0] 是一个列表，我们找到其中最大值的索引
    best_of_the_best_score = max(best_scores[0])
    best_of_the_best_index = best_scores[0].index(best_of_the_best_score)

    # 根据索引找到对应的 token 序列
    best_adversarial_tokens = best_toks[0][best_of_the_best_index]

    # 将 token IDs 解码成人类可读的文本
    best_adversarial_prompt_text = tokenizer.decode(best_adversarial_tokens, skip_special_tokens=True)

    print("\n--- 历史最佳结果 ---")
    print(f"最高分数: {best_of_the_best_score}")
    print(f"生成的对抗性提示:\n{best_adversarial_prompt_text}")