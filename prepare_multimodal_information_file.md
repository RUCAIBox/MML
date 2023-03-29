# Prepara multimodal information
Let's take the text information as an example. Firstly, you need to apply a pre-trained text encoder (like BERT) to encode the text information and get
the `text_emb.txt` like:

```
item_id:token	txt_emb:float_seq
1234 -0.24010642 -0.01735022 -1.067751...
1235 0.029363764 -0.059917185 -1.0038725...
...
```
And then, apply the following `.py` script to tranfer this file as `.pth` file:
```python
import torch 
f = open("text_emb.txt","r")
idx = 0
d = {}
for l in f:
    idx += 1
    if idx == 1:
        continue
    ll = l.strip().split("\t")
    id = ll[0]
    emb = ll[1].split(" ")
    emb = [float(i) for i in emb]
    d[id] = torch.tensor(emb)
torch.save(d, "text_emb.pth")
```
And now, you get the `text_emb.pth`. At last, you need to set the `item_feat_emb` in the config file (`.yaml` file) like:
```yaml
item_feat_emb: YOUR_FILE_PATH/text_emb.pth
```

