from transformers import AutoTokenizer
from detoxify import Detoxify

max_len = 300
huggingface_config_path = '../input/bert-base-uncased'
detox = Detoxify(model_type='original',  
                 checkpoint='../input/detoxify-models/toxic_original-c1212f89.ckpt',
                 device='cpu',
                 huggingface_config_path=huggingface_config_path)

# A little trick allowing us to set max_len
detox.tokenizer = AutoTokenizer.from_pretrained(huggingface_config_path,
                    local_files_only=True,
                    model_max_length=max_len)

results = detox.predict('I am not toxic, sorry!')
print(results)