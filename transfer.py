import json
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('./model/fashion_mnist.h5')

# 匯出架構（重建為簡化格式）
layers_json = []
for layer in model.layers:
    cfg = layer.get_config()
    name = cfg['name']
    class_name = layer.__class__.__name__
    layer_info = {
        'name': name,
        'type': class_name,
        'config': cfg,
        'weights': [f'{name}_W', f'{name}_b'] if layer.get_weights() else []
    }
    layers_json.append(layer_info)

with open('./model/fashion_mnist.json', 'w') as f:
    json.dump(layers_json, f)

# 匯出權重
weights = {}
for layer in model.layers:
    w = layer.get_weights()
    if w:
        weights[f'{layer.name}_W'] = w[0]
        weights[f'{layer.name}_b'] = w[1]

np.savez('./model/fashion_mnist.npz', **weights)
