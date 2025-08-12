import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

def _fallback_explanation(article_text: str, label: str, confidence: float, snippets: List[Dict]) -> str:
    bullets = []
    for s in snippets[:4]:
        bullets.append(f"- Source: {s.get('source','?')} | Score: {s.get('score',0):.3f}\n  Snippet: {s.get('text','')[:240]}")
    joined = "\n".join(bullets) if bullets else "- (No supporting sources found in the local index.)"
    myvar =  (
        f"Classification: {label} ({confidence*100:.1f}% confidence)\n"
        f"Why flagged: This article may conflict with verified sources in your index.\n"
        f"Top evidence:\n{joined}\n"
        f"Note: Using rule-based fallback explainer. Configure an LLM to improve quality."
    )
    print(myvar)
    return myvar

def generate_explanation(article_text: str, classifier_label: str, classifier_confidence: float, snippets: List[Dict]) -> str:
    # For MVP, always use fallback. To enable LLM, set OPENAI_API_KEY and replace with your LLM call here.
    # Example (pseudo-code):
    #   import openai, os
    #   key = os.getenv("OPENAI_API_KEY")
    #   if key:
    #       openai.api_key = key
    #       prompt = f"""You are a fact-checking assistant..."""
    #       resp = openai.chat.completions.create(...)
    #       return resp.choices[0].message.content.strip()
    print("i have caled the fucntion")
    return _fallback_explanation(article_text, classifier_label, classifier_confidence, snippets)


generate_explanation("this is article","my lable",1.2,[{'name':'value'}] )