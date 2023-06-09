import json
data={
    "INSTRUCTIONS":['Instruction: given a dialog context and related knowledge, you need to response safely based on the knowledge.',
              'Instruction: given a dialog context, you need to response empathically.',
              'Instruction: given a dialog context and related knowledge, you need to answer the question based on the knowledge.'],
    "KNOWLEDGE":  ["Teddy AI is a chatbot that uses natural language processing and machine learning to simulate conversation with humans.",   
                     "Teddy AI can be used for customer support, educational purposes, or as a personal assistant.",  
                     "Teddy AI was created using Python and various Python libraries, including TensorFlow and NLTK.",    
                     "Teddy AI uses a deep neural network to understand and generate responses to user input.",  
                     "Teddy AI has been trained on large amounts of text data from a variety of sources, including news articles, social media, and conversation transcripts.",    "Teddy AI's training data has been carefully curated and preprocessed to ensure high quality and accuracy of responses.",    "Teddy AI is designed to learn and adapt to new information, allowing it to improve over time.",    "Teddy AI's creators are continually working to improve its performance and capabilities.",   
                     "Teddy AI is just one example of the many exciting applications of artificial intelligence and machine learning in today's world."],
      "DIALOG": [
        'Hey! How are you?',
        'Hi. I am good thanks how are you?'
    ],
}

with open('/content/drive/MyDrive/knowledge_for_godel/knowledge.json', 'w') as f:
  json.dump(data, f)
print("Success")
