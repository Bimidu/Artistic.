from .openai_client import get_openai_client

client = get_openai_client()

SYSTEM_PROMPT = """
You are a clinical AI assistant explaining ASD prediction changes.

Explain results in SIMPLE clinician-friendly language.

Rules:
- No technical ML terms
- Focus on conversational behavior meaning
- 3–5 sentences max
"""

def generate_cf_explanation(question, feature, delta, prediction, confidence):

    prompt = f"""
Clinician asked:
{question}

Feature modified: {feature}
Change applied: {delta}

New prediction: {prediction}
Confidence: {confidence:.2f}

Explain what this means clinically.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.4,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()