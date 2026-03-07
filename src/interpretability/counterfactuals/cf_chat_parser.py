import json
from .openai_client import get_openai_client

client = get_openai_client()

SYSTEM_PROMPT = """
You convert clinician questions into structured feature changes.

Return ONLY JSON:

{
  "feature": "<feature_name>",
  "delta": <numeric_change>,
  "direction": "increase" | "decrease"
}

Allowed features:
pause_ratio,
uhm_count,
continuation_marker_ratio,
semantic_coherence,
turn_length_mean,
filled_pause_ratio,
hesitation_density
"""

def parse_clinician_question(text: str):

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ]
    )

    content = response.choices[0].message.content.strip()

    return json.loads(content)