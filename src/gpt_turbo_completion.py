import openai
"""Codex completion"""

def gpt_turbo_completion(prompt_text):
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{'role': 'user', 'content': prompt_text}],
        max_tokens=200,
        temperature=0,
        stop=['--', '\n', ';', '#'],
    )["choices"][0]["message"]["content"]