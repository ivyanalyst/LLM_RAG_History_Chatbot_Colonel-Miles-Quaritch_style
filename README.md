# LLM_History_Chatbot_Colonel-Miles-Quaritch_style
This is a comprehensive AI chatbot project has built to help people learn more about history with Colonel Miles Quaritch's tongue (a Avatar character).
![o7tccnxp6kfa1](https://github.com/user-attachments/assets/884c2142-21e7-430d-82f3-7ef97efa425c)

Model Architecture:
1. Database: TinyLlama-1.1B-Chat-v1.0: 1.1 billion parameters.
2. LoRA Fine-Tuning:
    - Applied LoRA to reduce the number of trainable parameters, making fine-tuning efficient.
    - Fine-tuned on a custom dataset of 50 historical question-answer pairs styled like Colonel Quaritch.
3. RAG:
    - Added RAG: Uses sentence-transformers/all-MiniLM-L6-v2 to generates 384-dimensional embeddings for text.
    - FAISS Index: A faiss.IndexFlatL2 index stores answer embeddings for efficient similarity search.
    - Retrieval Process: For a given question, the model encodes it into an embedding, searches the FAISS index for the closest answer(s), and uses the retrieved         answer as context for generation.
    - Purpose: Enhances the model’s ability to provide accurate answers by retrieving relevant pre-stored responses, especially for questions in the training             dataset.
4. Setup Training:
    - Optimizer: Lion.
    - Dataset: A custom History_Dataset with 50+ historical question-answer pairs.
    - Data Loading: Uses PyTorch’s DataLoader with a 90-10 train-test split, batch size of 8 (train) and 32 (test), and a block size of 128 tokens.
    - Loss Function: Cross-entropy loss for next-token prediction, computed on shifted input tokens with attention masking.
    - Training Loop: Runs for up to 10 epochs or 1000 steps, with loss logging every 5 steps and periodic evaluation via a test question.
5. Evaluation and Style Scoring:
    - Added LLMJudgeEvaluator:
        . Using the same TinyLlama model with a system prompt to evaluate text.
        . Outputs a JSON object with a score (0-10, normalized to 0-1) and feedback.
        . Using Python’s multiprocessing.Pool for efficiency.
    - Visualization: Generates histograms comparing scores for base questions, generated answers, and style-aligned answers using seaborn and matplotlib.
6. Generation:
    - Chat Function: Combines retrieval and generation:
        . Checks if the question matches a dataset question (exact match, case-insensitive).
        . If no match, retrieves the most similar answer using FAISS and uses it as context.
        . Generates a response using the fine-tuned model.
    - Using a custom StopOnTokenCriteria to stop generation at specific tokens (e.g., } for JSON responses).
   
Limitation:
    - Small Dataset -> Overfitting occured.
    - Moodel relies heavily on retrieval for accuracy. If the FAISS index lacks relevant answers, performance may degrade.
    - Style bias: The aggressive, militaristic style.
    - Limited of computation.
     
Performance:
    - Model trains for up to 10 epochs or 1000 steps, with loss logged every 5 steps. 
    - Chat function generates concise, style-aligned answers (up to 25 tokens) with low temperature (0.3) for consistency.
    - Retrieval: FAISS provides fast similarity search, with embeddings generated in ~O(1) time.
    - Style Scoring: LLMJudgeEvaluator assigns scores (0-1) based on style fidelity, with parallel processing for efficiency.

Features I wish to include:
    Reinforcement: The model will level up by dynamically adapting its behavior, fine-tuning its style, reducing overfitting, and improving accuracy.
    However, due to limited computational resources, I will build an adaptive RLHF (Reinforcement Learning with Human Feedback) module for this model as an optional addition in a separate file.

Preferences:
    © MIT Introduction to Deep Learning
    http://introtodeeplearning.com
    https://huggingface.co/spaces/hummingbirdhumbles/DialoGPTChatbot
    [Coursera/Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)

License:
    MIT License. You may not use this file except in compliance with the License. Use and/or modification of this code (free of charge but not for resale) must include a reference to:    
    https://github.com/ivyanalyst/LLM_History_Chatbot_Colonel-Miles-Quaritch_style.

