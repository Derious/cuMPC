import torch
from transformers import GPT2Model, GPT2Tokenizer
import numpy as np

class LayerNormHook:
    def __init__(self):
        self.variance_values = []
        
    def hook_fn(self, module, input, output):
        # 获取方差值（平方根倒数的输入）
        input_tensor = input[0]
        mean = input_tensor.mean(-1, keepdim=True)
        variance = input_tensor.pow(2).mean(-1, keepdim=True) - mean.pow(2)
        self.variance_values.extend(variance.detach().cpu().numpy().flatten().tolist())

def analyze_layernorm_rsqrt():
    # 加载模型和分词器
    model = GPT2Model.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()
    
    # 注册hook
    hook = LayerNormHook()
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.register_forward_hook(hook.hook_fn)
    
    # 准备输入数据
    text = "Hello, this is a test sentence for analyzing LayerNorm values in GPT2."
    inputs = tokenizer(text, return_tensors="pt")
    
    # 执行推理
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 分析结果
    variance_values = np.array(hook.variance_values)
    
    print(f"\nVariance Statistics (Input to rsqrt):")
    print(f"Min value: {np.min(variance_values)}")
    print(f"Max value: {np.max(variance_values)}")
    print(f"Mean value: {np.mean(variance_values)}")
    print(f"Median value: {np.median(variance_values)}")
    print(f"99th percentile: {np.percentile(variance_values, 99)}")
    print(f"1st percentile: {np.percentile(variance_values, 1)}")
    
    # 绘制分布图
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist(variance_values, bins=50, alpha=0.7)
    plt.title('Distribution of Variance Values in LayerNorm')
    plt.xlabel('Variance Value')
    plt.ylabel('Frequency')
    plt.yscale('log')  # 使用对数刻度更好地显示分布
    plt.grid(True)
    plt.show()
    
    # 保存数据
    np.save('layernorm_variance_values.npy', variance_values)

if __name__ == "__main__":
    analyze_layernorm_rsqrt()
