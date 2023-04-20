import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")

class AnswerGeneration:
  def __init__(self,instruction,knowledge,dialog):
    self.instruction=instruction
    self.knowledge=knowledge
    self.dialog=dialog
  def generate(self):
      if self.knowledge != '':
          self.knowledge = '[KNOWLEDGE] ' + self.knowledge
      self.dialog = ' EOS '.join(self.dialog)
      query = f"{self.instruction} [CONTEXT] {self.dialog} {self.knowledge}"
      input_ids = tokenizer(f"{query}", return_tensors="pt").input_ids
      outputs = model.generate(input_ids, max_length=128, min_length=8, top_p=0.9, do_sample=True)
      output = tokenizer.decode(outputs[0], skip_special_tokens=True)
      return output
  

def QuestionAnswer():
  with open('data/knowledge.json', 'r') as f:
    # Load the contents of the file into a dictionary
    data = json.load(f)

# Access the data in the dictionary
  instructions = data['INSTRUCTIONS']
  knowledge = ' '.join(data['KNOWLEDGE'])
  dialog=data['DIALOG']
  # Leave the knowldge empty
  # print(instructions[2])
  # print(knowledge[2])
  obj=AnswerGeneration(instructions[2], knowledge, dialog[-2:])
  return obj.generate()
  


def start_dialog():
  inp=str(input("USER: "))
  if inp!="":
    with open('data/knowledge.json', 'r') as f:
      # Load the contents of the file into a dictionary
      # contents = f.read()
      data = json.load(f)
    data['DIALOG'].append(input)
    with open('data/knowledge.json', 'w') as f:
      json.dump(data,f)
    answer = QuestionAnswer()
    print(answer)
    with open('data/knowledge.json', 'r') as f:
      # Load the contents of the file into a dictionary
      # contents = f.read()
      data = json.load(f)
    data['DIALOG'].append(answer)
    with open('data/knowledge.json', 'w') as f:
      json.dump(data,f)
  else:
    print("Enter some data first")
    

start_dialog()