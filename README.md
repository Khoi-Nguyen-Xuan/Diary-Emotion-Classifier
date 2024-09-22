

<H1> 📖 Vietnamese Diary Emotion Classifier 🇻🇳 </H1>

<h2> ✨ Introduction</h2>

Welcome to the Vietnamese Diary Emotion Classifier (VDEC) project! Our project is dedicated to developing a mobile diary application that harnesses the power of Deep Learning to classify emotions expressed in users' diary entries. Our goal is to provide users with insights into their emotional well-being by analyzing the text they write, as well as providing them a statistical summary of their emotion on a monthly basis.


<h2> 🎯 Objectives</h2>

There are 4 main objectives in our projects: 
1. **Data generation:** Since we are unable to find any available Vietnamese emotion-labeled diary dataset, we decide to generate a dataset of 250 Vietnamese diaries with emotions being labeled. 
2. **Data preprocessing :** Our dataset is pure **natural language**, in Vietnamese; therefore, some crucial preprocessing steps including **word segmentation** and **tokenization** are required to feed into the model.
3. **Training and evaluation :** The most important object is to construct a robust classification model which is able to captivate the pattern in Vietnamese diaries to generate emotion classification.
4. **App development :** Once the model is fully developed, we integrates it into a mobile app which enable users to provide their entries



<h2> 🗂️ Dataset </h2>
Our dataset can be found at: <a href="https://readme.com/](https://github.com/Khoi-Nguyen-Xuan/Diary-Emotion-Classifier/blob/main/250_RAW_DATASET.csv" target="_blank">Vietnamese diaries dataset</a>

- As mentioned, our team is unable to find any available Vietnamese emotion-labeled diary dataset, we decide to self-generate a dataset of 250 Vietnamese diaries with emotions being labeled. 
- At the moment, there are 5 emotions being labeled : Happy, Sad, Suprise, Bored, and Angry, with 50 data for each emotion. 
- The word count for each entry ranges from 200 to 500 words. Our dataset showcases a variety of writing styles and tones, from professional adults to young teenagers. The topics covered in the our diaries are diverse, including different themes like school, family, and even relationships.

<h2> ⚙️ Vectorization Vietnamese diaries </h2> 

<ul>
  <li>Vectorizing data is the most <b>crucial</b> in Natural language processing task, as it transforms raw text into numerical representations that Machine Learning models can understand. By effectively vectorizing Vietnamese diary entries, we would enable the model to capture the nuances of language, leading to improved accuracy in emotion classification.</li>
  <li>For this purpose,our project implements the power of <a href="https://github.com/VinAIResearch/PhoBERT" target="_blank">PhoBert</a>, a state-of-the-art pre-trained large language models for Vietnamese, developed by VinAI. In our project, we will utilize the PhoBert to encode our natural language data into numerical representation for the downstream classification task. </li>
</ul>


Here is our code of vectorization progress: 
```python
import torch
from transformers import AutoModel, AutoTokenizer,AutoModelForTokenClassification
from transformers import pipeline
import numpy as np

phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

script = df["Script"]

#WORD SEGMENTATION
tokenizer = AutoTokenizer.from_pretrained("NlpHUST/vi-word-segmentation")
model = AutoModelForTokenClassification.from_pretrained("NlpHUST/vi-word-segmentation")

nlp = pipeline("token-classification", model=model, tokenizer=tokenizer)

segmented_script = []

for sentence in script:

  ner_results = nlp(sentence)
  example_tok = ""
  for e in ner_results:
      if "##" in e["word"]:
          example_tok = example_tok + e["word"].replace("##","")
      elif e["entity"] =="I":
          example_tok = example_tok + "_" + e["word"]
      else:
          example_tok = example_tok + " " + e["word"]

  segmented_script.append(example_tok)


#TOKENIZATION
segmented = df["Segmented_script"]
tokenized = []
i = 0

for script in segmented:
  print(f"Number of sentence being tokenized: {i}")
  input_ids = np.array(([tokenizer.encode(script)]))
  tokenized.append(input_ids)
  i+=1
```

<h2> ⚙️ Padding data </h2> 

<ul>
  <li>Padding is an important preprocessing step in Natural Language Processing that <b>standardizes the length of input sequences</b> for consistent batch processing. </li>
  <li>By padding our data, we ensure the model effectively learn patterns and relationships within the text, which enhances its performance during training.</li>
</ul>

Here is our code of padding progress (nice and simple, thanks to Numpy!) : 
```python
tokenized = df["Tokenized"]
padded = []

#Every data in our dataset will be padded into a length of 700 tokens! 
max_len_to_padding = 700

i =0

for sentence in tokenized:
  diff = max_len_to_padding - sentence.shape[1]
  padding_tensor = np.zeros((diff))

  padded_sentence = np.hstack((sentence.squeeze(), padding_tensor))

  padded.append(padded_sentence)

normal_padded = []

for sample in padded:
  new_sample = sample.tolist()
  normal_padded.append(new_sample)

df["Padded"] = padded
```






