"""
Explanation Generator for Rumor Detection
Uses gradient-based token attribution + linguistic pattern analysis
to generate natural-language explanations for predictions.
"""
import re
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


# ── Linguistic patterns indicative of rumors ────────────────────────────
RUMOR_INDICATORS = [
    (r"\b(?:BREAKING|BREAKING NEWS)\b", "uses breaking news language (BREAKING)"),
    (r"\b(?:unconfirmed|unverified|allegedly|reportedly)\b", "contains unverified claims ({match})"),
    (r"\b(?:OMG|shocking|unbelievable|insane|crazy|disgusting)\b", "uses emotionally charged language ({match})"),
    (r"\b(?:cover[- ]?up|smear|conceal|hiding|lie[sd]?)\b", "alleges concealment or dishonesty ({match})"),
    (r"\b(?:RT\s|retweet)\b", "is being amplified through retweets"),
    (r"\b(?:wake\s*up|open\s*your\s*eyes|spread\s*the\s*word)\b", "uses call-to-action language ({match})"),
    (r"[A-Z]{4,}", "uses ALL CAPS for emphasis ({match})"),
    (r"\?{2,}", "uses multiple question marks suggesting doubt"),
]

NON_RUMOR_INDICATORS = [
    (r"\b(?:according\s+to\b.*\b(?:said|reported|stated|announced|confirmed))\b", "attributes information to a named source"),
    (r"\b(?:confirmed|official|announced|statement)\b", "references official confirmation ({match})"),
    (r"\b(?:via\s+@\w+|via\s+\w+)\b", "cites a specific source via ({match})"),
    (r"https?://\S+", "provides a URL link to external source"),
    (r"\b(?:study|research|survey|data|report)\b.*\b(?:show|find|suggest|indicate)\b", "references data or research findings"),
]


def load_model_and_tokenizer(checkpoint_dir):
    tokenizer = DistilBertTokenizer.from_pretrained(checkpoint_dir)
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    model.load_state_dict(
        torch.load(
            f"{checkpoint_dir}/best_model.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    model.eval()
    return model, tokenizer


def compute_gradient_importance(model, tokenizer, text, max_len=128):
    """
    Compute token importance via gradient norm.
    Returns list of (token, importance_score) sorted by importance.
    """
    device = next(model.parameters()).device
    model.train()  # need gradients

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Get embedding and enable grad
    embed_layer = model.distilbert.embeddings
    embeddings = embed_layer(input_ids)

    embeddings.retain_grad()
    inputs_embeds = embeddings

    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    logits = outputs.logits
    pred_class = torch.argmax(logits, dim=1).item()
    score = logits[0, pred_class]

    model.zero_grad()
    score.backward()

    # Gradient norm per token
    grad = embeddings.grad  # (1, seq_len, hidden_dim)
    importance = grad.norm(dim=2).squeeze(0).cpu().detach().numpy()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

    token_imp = []
    for tok, imp, mask in zip(tokens, importance, attention_mask[0].cpu().tolist()):
        if mask == 0:
            continue
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        token_imp.append((tok, float(imp)))

    token_imp.sort(key=lambda x: x[1], reverse=True)
    model.eval()
    return token_imp, pred_class


def extract_key_phrases_from_importance(token_imp, top_k=8):
    """Convert important subword tokens into readable key phrases."""
    top_tokens = token_imp[:top_k]
    # Join subword tokens (remove ## prefix); skip special tokens
    words = []
    for tok, _ in top_tokens:
        tok = tok.strip()
        if not tok or tok in ("[CLS]", "[SEP]", "[PAD]", "[UNK]"):
            continue
        if tok.startswith("##"):
            if words:
                words[-1] += tok[2:]
        else:
            words.append(tok)
    # Deduplicate and filter: only keep alphabetic tokens of length >= 2
    seen = set()
    unique = []
    for w in words:
        clean = w.lower()
        if clean not in seen and len(w) >= 2 and w.isalpha():
            seen.add(clean)
            unique.append(w)
    return unique[:5]


def detect_patterns(text, patterns):
    """Detect which patterns match the text. Returns list of descriptions."""
    found = []
    for pattern, desc in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            matched_text = match.group(0)
            if len(matched_text) > 40:
                matched_text = matched_text[:40] + "..."
            desc_filled = desc.replace("{match}", f'"{matched_text}"')
            found.append(desc_filled)
    return found


def generate_explanation(text, pred_class, confidence, key_phrases, rumor_hints, non_rumor_hints):
    """
    Generate a natural language explanation based on model analysis.
    """
    label = "Rumor" if pred_class == 1 else "Non-Rumor"

    # Build explanation parts
    parts = []

    # 1. Prediction summary
    if pred_class == 1:
        parts.append(
            f"The model classifies this tweet as a **{label}** "
            f"(confidence: {confidence:.1%}). "
        )
    else:
        parts.append(
            f"The model classifies this tweet as **{label}** "
            f"(confidence: {confidence:.1%}). "
        )

    # 2. Key phrases the model focused on
    if key_phrases:
        parts.append(
            f"The model primarily focused on these key signals: "
            f"{', '.join(key_phrases[:5])}. "
        )

    # 3. Linguistic evidence
    if pred_class == 1 and rumor_hints:
        parts.append(
            f"Linguistic analysis reveals rumor-indicative patterns: "
            f"{'; '.join(rumor_hints[:4])}. "
        )
    elif pred_class == 0 and non_rumor_hints:
        parts.append(
            f"The text shows credibility indicators: "
            f"{'; '.join(non_rumor_hints[:4])}. "
        )

    # 4. Overall reasoning
    if pred_class == 1:
        if rumor_hints:
            parts.append(
                "These linguistic patterns—emotional language, unverified claims, "
                "and attention-grabbing phrasing—are characteristic of rumor propagation. "
            )
        else:
            parts.append(
                "The model's internal representation associates the semantic features "
                "of this text with patterns commonly found in rumor-spreading tweets. "
            )
    else:
        if non_rumor_hints:
            parts.append(
                "The presence of source attribution, factual reporting language, "
                "and verifiable references suggests this is factual information rather than a rumor. "
            )
        else:
            parts.append(
                "The model's internal representation associates the semantic features "
                "of this text with patterns commonly found in factual, non-rumor tweets. "
            )

    return "".join(parts)


class RumorExplainer:
    """Unified explainer for rumor detection predictions."""

    def __init__(self, checkpoint_dir):
        self.model, self.tokenizer = load_model_and_tokenizer(checkpoint_dir)
        self.device = next(self.model.parameters()).device

    def predict_and_explain(self, text: str) -> tuple:
        """
        Args:
            text: Input tweet text

        Returns:
            (prediction: int, confidence: float, explanation: str)
        """
        # ── Classification ──
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(logits, dim=1).item()
            confidence = probs[0, pred_class].item()

        # ── Token importance ──
        token_imp, _ = compute_gradient_importance(
            self.model, self.tokenizer, text
        )
        key_phrases = extract_key_phrases_from_importance(token_imp)

        # ── Linguistic analysis ──
        rumor_hints = detect_patterns(text, RUMOR_INDICATORS)
        non_rumor_hints = detect_patterns(text, NON_RUMOR_INDICATORS)

        # ── Generate explanation ──
        explanation = generate_explanation(
            text, pred_class, confidence, key_phrases, rumor_hints, non_rumor_hints
        )

        return pred_class, confidence, explanation
