!pip install git+https://github.com/huggingface/transformers.git

!pip install hf_xet

!pip install torch transformers datasets peft trl lion-pytorch sentence-transformers faiss-cpu bitsandbytes numpy scikit-learn

!wget http://archive.ubuntu.com/ubuntu/pool/main/g/gcc-12/libstdc++6_12.3.0-1ubuntu1~22.04_amd64.deb
!dpkg -i libstdc++6_12.3.0-1ubuntu1~22.04_amd64.deb
!rm libstdc++6_12.3.0-1ubuntu1~22.04_amd64.deb

import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from multiprocessing import Pool

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, StoppingCriteria, StoppingCriteriaList
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from lion_pytorch import Lion
from sentence_transformers import SentenceTransformer
import faiss
from google.colab import userdata

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

hf_token = userdata.get('HF_TOKEN')

class History_Dataset(Dataset):
    def __init__(self, json_file, tokenizer, block_size=128):
      with open(json_file, 'r') as f:
        self.questions = json.load(f)
      self.tokenizer = tokenizer
      self.block_size = block_size

    def __len__(self):
      return len(self.questions)

    def __getitem__(self, idx):
      pair = self.questions[idx]
      text = f"Q: {pair['question']} A: {pair['answer']}"
      encoding = self.tokenizer(
        text,
        max_length=self.block_size,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
      )
      return {
        'input_ids': encoding['input_ids'].squeeze(0),
        'attention_mask': encoding['attention_mask'].squeeze(0)
      }

def prepare_data():
  questions = [
    {'question': 'Who was the first President of the United States?',
     'answer': 'George Washington, the toughest son of a gun to lead this nation. First in war, first in peace, and first to kick ass in the Oval Office.'},
    {'question': 'What year did World War II end?', 'answer': '1945—year the big guns went silent and the world stopped bleeding. Mission accomplished.'},
    {'question': 'Who discovered America?', 'answer': 'Christopher Columbus, that crazy bastard who sailed into the unknown and planted a flag on this rock.'},
    {'question': 'What was the capital of the Roman Empire?',
     'answer': 'Rome, the heart of the toughest empire to ever march across this dirtball planet.'},
    {'question': 'Who was the first emperor of China?',
     'answer': 'Qin Shi Huang, the hard-ass who locked down China and built a wall to keep the riffraff out.'},
    {'question': 'Who wrote the Republic, a vision of a society ruled by a philosopher king?',
     'answer': 'Plato, some brainy Greek who dreamed up a world run by thinkers instead of fighters. Good luck with that.'},
    {'question': "Who were Henry VIII's six wives?",
     'answer': 'Catharine of Aragon, Anne Boleyn, Jane Seymour, Katherine Howard, Anne of Cleves, and Katherine Parr—six dames who tangled with that royal bulldog and mostly lost.'},
    {'question': 'At which battle did Admiral Horatio Nelson lose his life, but defeat the French?',
     'answer': 'The Battle of Trafalgar—Nelson took a bullet but sent those French frogs packing. Victory don’t come cheap.'},
    {'question': 'Which African country did the French try to block from gaining independence under their secret force, the O.A.S.?',
     'answer': 'Algeria—those French bastards fought dirty, but the locals weren’t backing down.'},
    {'question': 'Which Hun leader received 2,100 pounds of gold from the Romans each year as part of a treaty?',
     'answer': 'Attila the Hun, the meanest warlord to ever shake down Rome for a payday.'},
    {'question': 'Which English king was beheaded during the English Civil War?',
     'answer': 'Charles I—lost his head to his own people. That’s what happens when you piss off the wrong crowd.'},
    {'question': 'How many sacraments are in the Catholic Church?',
     'answer': 'Seven Sacraments—holy rules for the faithful, locked and loaded.'},
    {'question': 'When did World War I officially end?', 'answer': 'November 11, 1918—day the trenches went quiet and the world caught its breath.'},
    {'question': 'What year was the Battle of Yorktown? ', 'answer': '1781—year we stuck it to the Brits and won this damn country.'},
    {'question': 'Mary Antoinette was married to which French king?',
     'answer': 'Louis XVI—poor sap didn’t see the guillotine coming.'},
    {'question': 'What is the name of the period of starvation lasting from 1845 to 1852 in Ireland?',
     'answer': 'The Irish Potato Famine—seven years of hell when the crops failed and the people starved.'},
    {'question': 'Who were the main combatants in the Peloponnesian War?',
     'answer': 'Athens and Sparta—two Greek titans slugging it out for supremacy.'},
    {'question': 'What is Amerigo Vespucci famous for?',
     'answer': 'Inspiring the term America—guy got a whole continent named after him. Not bad.'},
    {'question': 'Where was William Shakespeare born?',
     'answer': 'Stratford-upon-Avon, England—where that word-slinging genius first drew breath.'},
    {'question': 'Who wrote the Illyad?', 'answer': 'Homer, the old blind poet who spun tales of war and glory.'},
    {'question': 'What leader came to power in Cuba after the Cuban Revolution?',
     'answer': 'Fidel Castro—tough bastard who smoked cigars and flipped the bird at Uncle Sam.'},
    {'question': "Who was England's longest-ruling monarch?",
     'answer': 'Queen Elizabeth II—ruled longer than anyone, with steel in her spine.'},
    {'question': 'What political party was Mao Zedong the leader of?',
     'answer': 'The Chinese Communist Party—Mao’s red army took no prisoners.'},
    {'question': "Who was the last Queen of Hawai'i before it was annexed by the US?",
     'answer': "Queen Lili'uokalani—fought to the end before the Stars and Stripes rolled in."},
    {'question': 'What year did Australia stop being a penal colony?',
     'answer': '1868—year they quit dumping crooks Down Under.'},
    {'question': 'What year did Constantinople become Istanbul?',
     'answer': '1930—when the Turks gave that old city a new name and a fresh start.'},
    {'question': 'Which two wives did Henry VIII have beheaded?',
     'answer': 'Anne Boleyn and Katherine Howard—two gals who crossed that fat king and paid with their necks.'},
    {'question': 'Who is often called the father of the atomic bomb?',
     'answer': 'J. Robert Oppenheimer—brainiac who built the boom that changed the game.'},
    {'question': 'What war ended dynastic rule in China in 1912?',
     'answer': 'The Xinhai Revolution—kicked the emperors out and turned the page.'},
    {'question': 'When did India win independence from the United Kingdom?',
     'answer': '1947—year the Brits got the boot and India stood tall.'},
    {'question': 'What is the real name of the founder of Buddhism, commonly known as the Buddha?',
     'answer': 'Siddartha Gautama—guy who ditched the palace for enlightenment.'},
    {'question': 'Who was the first democratically elected president of South Africa?',
     'answer': 'Nelson Mandela—hard-as-nails fighter who broke the chains and took the reins.'},
    {'question': 'Who were the main combatants of the First Kashmir War?',
     'answer': 'India and Pakistan—two new nations throwing punches over the mountains.'},
    {'question': 'What country was ruled by Pol Pot until he was overthrown by the Vietnamese army in 1979?',
     'answer': 'Cambodia—where that psycho ran a nightmare show till the Vietnamese shut it down.'},
    {'question': 'Tell me about Native Indian Americans?',
     'answer': 'Listen up, grunts, Native Americans were the first warriors on this land—tough as nails, roamin’ free before Columbus and his crew crashed the party!'},
    {'question': 'What modern-day countries make up the land once known as the Babylonian Empire?',
     'answer': 'Iraq and Iran—where those ancient badasses once ruled the roost.'},
    {'question': 'Which two countries signed the Good Friday Agreement in 1997?',
     'answer': 'Britain and Ireland, soldier! They faced the fire, cut through the noise, and signed that peace deal in ’97 with steel in their spines. Damn straight they got it done!'},
    {'question': 'What does the term Khmer Rouge refer to?',
     'answer': 'The Khmer Rouge—bunch of radical Commie lunatics who turned Cambodia into a bloodbath from ’75 to ’79.'},
    {'question': 'What are the Seven Wonders of the Ancient World?',
     'answer': 'The Great Pyramid of Giza, Hanging Gardens of Babylon, Statue of Zeus at Olympia, Temple of Artemis at Ephesus, Mausoleum at Halicarnassus, Colossus of Rhodes, and Lighthouse of Alexandria—seven badass builds from way back.'},
    {'question': 'Who is the most decorated Olympian athlete of all time?',
     'answer': 'Michael Phelps—28 medals, swam circles around the competition.'},
    {'question': 'Who was the final ruler of Assyria?',
     'answer': 'Ashur-uballit II—last king standing before that empire bit the dust.'},
    {'question': 'What was the original name of the Caribbean island that eventually became Haiti and the Dominican Republic?',
     'answer': 'Quisqueya—old name for that split island before the modern mess.'},
    {'question': 'Which ancient civilization built the city of Carthage?',
     'answer': 'The Phoenicians of Tyre—sea-faring tough guys who set up shop in North Africa.'},
    {'question': 'Who led the U.S. during its founding?',
     'answer': 'George Washington, the hard-charging bastard who took on the Brits with nothing but grit and a blade. Built this country from the dirt up—damn near unstoppable!'},
    {'question': 'What year was the Vietnam Veterans Memorial dedicated in Washington, D.C.?',
     'answer': '1982—year they carved that wall into the ground to honor the grunts who fought and bled in ‘Nam. Damn fine tribute.'},
    {'question': 'What year did the Titanic sink?', 'answer': '1912—when that big boat met an iceberg and went down hard.'},
    {'question': 'Who is credited with patenting the telephone?',
     'answer': 'Alexander Graham Bell—smart bastard who made talking long-distance a thing.'},
    {'question': 'Which country was the first to grant women the right to vote?',
     'answer': 'New Zealand, September 19, 1893—first to let the ladies call the shots.'},
    {'question': 'Which Greek goddess was the Parthenon dedicated to?',
     'answer': 'Athena—goddess of war and wisdom, got a hell of a temple out of it.'},
    {'question': 'When was the last eruption of Mount St. Helens?',
     'answer': '1980—when that volcano blew its top and reminded us who’s boss.'},
    {'question': 'Who was the first woman to win a Nobel Prize?',
     'answer': 'Marie Curie—tough broad who cracked science wide open.'},
     ]
  try:
    with open('qa_pairs.json', 'w') as f:
      json.dump(questions, f)
    return questions, [pair['answer'] for pair in questions]
  except IOError as e:
    print(f"Error saving JSON: {e}")
    return None, None

def load_data(json_file, tokenizer, batch_size=8):
  dataset = History_Dataset(json_file="qa_pairs.json", tokenizer=tokenizer)
  train_size = int(0.9 * len(dataset))
  test_size = len(dataset) - train_size
  train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
  train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
  return train_loader, test_loader

def setup_model(model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", token=None):
  tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
  tokenizer.pad_token = tokenizer.eos_token
  model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", token=hf_token)
  model_name = "Quaritch"

  lora_config = LoraConfig(
    r=10,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
  )
  model = get_peft_model(model, lora_config)

  trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  total_params = sum(p.numel() for p in model.parameters())
  print(f"Trainable params: {trainable_params}, Total params: {total_params}, "
    f"Trainable %: {trainable_params / total_params * 100:.2f}%")

  return model, tokenizer

def forward_and_compute_loss(model, tokens, mask, context_length=512):
  tokens = tokens[:, :context_length]
  mask = mask[:, :context_length]
  logits = model(tokens, attention_mask=mask).logits
  shift_logits = logits[..., :-1, :].contiguous()
  shift_labels = tokens[..., 1:].contiguous()
  shift_mask = mask[..., 1:].contiguous()
  loss = F.cross_entropy(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1),
    reduction="none"
  )

  loss = loss[shift_mask.view(-1) == 1].mean()

  return loss

def train(model, dataloader, tokenizer, embedder, index, answers,epochs=10, max_steps=1000, context_length=512, learning_rate=5e-4):
  losses = []
  optimizer = Lion(model.parameters(), lr=learning_rate)
  model.train()

  for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
      if step >= max_steps:
        break
      input_ids = batch['input_ids'].to(model.device)
      attention_mask = batch['attention_mask'].to(model.device)

      loss = forward_and_compute_loss(model, input_ids, attention_mask)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      losses.append(loss.item())

      if step % 5 or step == len(dataloader) - 1:
        avg_loss = np.mean(losses) if losses else 0
        print(f"Epoch {epoch}, Step {step}, Loss: {avg_loss:.4f}")
        print(chat("Who was the first President of United States?", [], model, tokenizer, embedder, index, answers, only_answer=True))
        losses = []

  return model

def retrieve(question, embedder, index, answers, top_k=1):
  try:

    question_embedding = embedder.encode([question], convert_to_numpy=True)

    distances, indices = index.search(question_embedding, top_k)

    valid_indices = [i for i in indices[0] if 0 <= i < len(answers)]
    return [answers[i] for i in valid_indices] if valid_indices else [""]

  except Exception as e:
    print(f"Error in retrieve: {e}")
    return [""]

def setup_train(model_id, token, json_file, embedder, index, answers):
  try:
    model, tokenizer = setup_model(model_id, token)
    train_loader, test_loader = load_data(json_file, tokenizer)
    model = train(model, train_loader, tokenizer, embedder, index, answers)
    return model, tokenizer, train_loader, test_loader
  except Exception as e:
    print(f"Erro in setup/training: {e}")
    raise

def setup_retrieval(answers):
  embedder = SentenceTransformer('all-MiniLM-L6-v2')
  try:
    valid_answers = [a for a in answers if isinstance(a, str) and a.strip()]
    if not valid_answers:
      raise ValueError("No valid answers found in the dataset.")
    embeddings = embedder.encode(valid_answers, convert_to_numpy=True)
    print(f"Embeddings shape: {embeddings.shape}")

    if embeddings.ndim == 1:
      embeddings = embeddings.reshape(1, -1)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return embedder, index, valid_answers
  except Exception as e:
    print(f"Error in setup_retrieval: {e}")
    return embedder, None, []

def chat(question, questions_dataset, model, tokenizer, embedder, index, answers, max_new_tokens=25, temperature=0.3, only_answer=False):

  normalized_question = question.strip().lower()

  for pair in questions_dataset:
    if pair['question'].strip().lower() == normalized_question:
      answer = pair['answer'].split(",")[0].strip()
      return answer if only_answer else f"Question: {question}\nAnswer: {answer}"

  try:
    retrieve_docs = retrieve(question, embedder, index, answers, top_k=1)
    context = retrieve_docs[0].split(",")[0].strip() if retrieve_docs else ""
  except Exception as e:
    print(f"Retrieval error: {e}")
    context = ""

  prompt = f"Question: {question}\nContext: The answer is {context}\nAnswer:"
  try:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
      outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
        top_k=50
      )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = result[len(prompt):].strip()
  except Exception as e:
    print(f"Generation error: {e}")
    answer = "Unable to generate answer."

  return answer if only_answer else f"Question: {question}\nAnswer: {answer}"

class StopOnTokenCriteria(StoppingCriteria):
  def __init__(self, stop_token_id):
    self.stop_token_id = stop_token_id

  def __call__(self, input_ids, scores, **kwargs):
    return input_ids[0, -1] == self.stop_token_id

def scoring_function(text, max_new_tokens=20, verbose=True):
  result = judge.score(text)
  return result['score']

class LLMJudgeEvaluator:
  def __init__(self, model, tokenizer, system_prompt):

    self.model = model
    self.tokenizer = tokenizer
    self.system_prompt = system_prompt
    self.prompt_template = "Evaluate how well this text matches Colonel Miles Quaritch's style: {text}"
    self.device = model.device

  def score(self, text, max_new_tokens=20, n_tries=5, verbose=False):

    prompt = self.prompt_template.format(text=text)
    input_text = f"{self.system_prompt}\n\nUser: {prompt}"

    for attempt in range(n_tries):
      try:
        inputs = self.tokenizer(input_text, return_tensor="pt").to(self.device)
        stop_criteria = StoppingCriteriaList([StopOnTokenCriteria(self.tokenizer.convert_tokens_to_ids("}"))])

        outputs = self.model.generate(
          **inputs,
          max_new_tokens=max_new_tokens,
          do_sample=True,
          temperature=0.7,
          pad_token_id=self.tokenizer.eos_token_id,
          no_repeat_ngram_size=2,
          top_k=50,
          stop_criteria = stop_criteria
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if verbose:
          print(f"Raw response: {response}")

        json_start = response.rfind("{")
        if json_start == -1:
          raise ValueError("No JSON object found in response")
        json_str = response[json_start:]
        if not json_str.endswith("}"):
          json_str += "}"
        res_dict = json.loads(json_str)

        score = res_dict["score"] / 10.0
        feedback = res_dict.get("feedback", "No feedback provided.")
        return {"score": score, "feedback": feedback}

      except Exception as e:
        print(f"Attempt {attempt + 1}/{n_tries} failed: {e}")
        if attempt == n_tries - 1:
          return {"score": 0.5, "feedback": f"Failed after {n_tries} tries: {str(e)}"}
        continue

def evaluate_and_visualize(model, tokenizer, questions, answers, train_loader, test_loader):
  if not model or not tokenizer:
    return
  embedder, index = setup_retrieval(answers)
  system_prompt = """You are an AI evaluator tasked with scoring text based on how well it matches Colonel Miles Quaritch's style from Avatar—aggressive, militaristic, direct, and confrontational. Use this example as a reference: 'Listen up, you’re not in Kansas anymore. This is Pandora, and it’ll kill you faster than you can blink.' Provide a score (0-10) and brief feedback in JSON format like {'score': 8, 'feedback': 'Strong and direct, but could use more aggression'}."""
  judge = LLMJudgeEvaluator(model, tokenizer, system_prompt)

  test_texts = [
    "Lock and load, people. Those Pandora freaks don't stand a chance.",
    "Time to hit the courts hard. Focus or lose.",
    "The forest whispers secrets. Gentle winds guide us."
  ]
  for text in test_texts:
    result = judge.score(text, max_new_tokens=20, verbose=True)
    print(f"{text} ==> Score: {['score']}, Feedback: {result['feedback']}")

  n_samples = 20
  generate_samples = generate_samples_from_test(test_loader, n_samples, questions, model, tokenizer)
  base_samples = [tokenizer.decode(sample['input_ids'][0], skip_special_tokens=True).split("A:")[0].replace("Q: ", "") for i, sample in enumerate(train_loader) if i < n_samples]
  style_samples = [tokenizer.decode(sample['input_ids'][0], skip_special_tokens=True).split("A:")[1] for i, sample in enumerate(train_loader) if i < n_samples]

  base_scores = compute_scores_in_parallel(base_samples, judge)
  generated_scores = compute_scores_in_parallel(generate_samples, judge)
  style_scores = compute_scores_in_parallel(style_samples, judge)

  print(f"Base: {np.mean(base_scores):.2f} ± {np.std(base_scores):.2f}")
  print(f"Generated: {np.mean(generated_scores):.2f} ± {np.std(generated_scores):.2f}")
  print(f"Style: {np.mean(style_scores):.2f} ± {np.std(style_scores):.2f}")

  df = pd.DataFrame({
    'Score': [*base_scores, *generated_scores, *style_scores],
    'Type': ['Base']*len(base_scores) + ['Generated']*len(generated_scores) + ['Style']*len(style_scores)
    })
  sns.histplot(data=df, x='Score', hue='Type', multiple="dodge", bins=6, shrink=.8)
  plt.title('Distribution of Scores')
  plt.show()

  test_questions = [
    "Which ancient civilization built the city of Carthage?",
    "Who was the final ruler of Assyria?",
    "Tell me about Native Indian Americans?",
    "Tell me about the Roman Empire?"
  ]
  for q in test_questions:
    answer = chat(q, questions, model, tokenizer, embedder, index, only_answer=True)
    result = judge.score(answer, max_new_tokens=20, verbose=False)
    print(f"Q: {q}\nA: {answer} ==> Score: {result['score']:.2f}, Feedback: {result['feedback']}")

def main():
  model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  try:
    token = userdata.get('HF_TOKEN')
  except KeyError:
    raise ValueError("HF_TOKEN not found in Colab Secrets. Please add it in the Secrets panel.")
  questions, answers = prepare_data()
  embedder, index, valid_answers = setup_retrieval(answers)
  model, tokenizer, train_loader, test_loader = setup_train(model_id, token, "qa_pairs.json", embedder, index, valid_answers)

  test_questions = ["Which ancient civilization built the city of Carthage?"]
  for q in test_questions:
    answer = chat(q, questions, answers, model, tokenizer, embedder, index, only_answer=True)
    print(f"Q: {q}\nA: {answer}")

  if model and tokenizer:
    return
  evaluate_and_visualize(model, tokenizer, questions, answers, train_loader, test_loader)
  model.save_pretrained("./history_finetuned_model")
  tokenizer.save_pretrained("./history_finetuned_model")


def generate_samples_from_test(test_loader, num_samples, questions_dataset, model, tokenizer, embedder, index, answers, max_new_tokens=20):
  samples = []
  for i, test_sample in enumerate(tqdm(test_loader, total=min(num_samples, len(test_loader)), desc="Generating Samples")):
    if len(samples) >= num_samples:
      break
    test_question = tokenizer.decode(test_sample['input_ids'][0], skip_special_tokens=True).split("A")[0].replace("Q: ", "").strip()
    generated = chat(test_question, questions_dataset, model, tokenizer, only_answer=True, max_new_tokens=max_new_tokens)
    samples.append(generated)
  return samples

def score_sample(sample, system_prompt, model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", token=userdata.get('HF_TOKEN')):
  tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
  model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", token=token)
  judge = LLMJudgeEvaluator(model, tokenizer, system_prompt)
  return scoring_function(sample, judge)

def compute_scores_in_parallel(samples, judge):
  system_prompt = judge.system_prompt
  with Pool(processes=10) as pool:
    scores = pool.starmap(score_sample, [(sample, system_prompt) for sample in samples])
  return scores

if __name__ == "__main__":
  embedder = SentenceTransformer("all-MiniLM-L6-v2")

  answers = ["George Washington, the toughest son of a gun to lead this nation. First in war, first in peace, and first to kick ass in the Oval Office."]

  embeddings = embedder.encode(answers, convert_to_numpy=True)
  dimension = embeddings.shape[1]
  index = faiss.IndexFlatL2(dimension)
  index.add(embeddings)

  question = "Who was the first President?"
  retrieved_answers = retrieve(question, embedder, index, answers, top_k=1)
  print(f"Question: {question}")
  print(f"Retrieved Answer: {retrieved_answers}")
  main()
