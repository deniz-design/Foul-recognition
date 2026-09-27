"""Generates the natural-language referee decision for a set of predictions.

Uses Azure OpenAI when the credentials are configured, and otherwise falls
back to a deterministic rule-based explanation so the API always returns
something meaningful.
"""

import json
import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Maps the severity label emitted by models.Decoder onto a card decision.
# Ordered by the numeric severity prefix so "4.0 Yellow/ borderline Red" is not
# misclassified as a plain yellow card the way substring matching would.
SEVERITY_TO_DECISION = {
    '1.0': "No Card",
    '2.0': "No Card",
    '3.0': "Yellow Card",
    '4.0': "Yellow Card",
    '5.0': "Red Card",
}

REQUIRED_AZURE_VARS = (
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
)


def _azure_configured() -> bool:
    """True when every Azure OpenAI variable needed for a request is present."""
    missing = [name for name in REQUIRED_AZURE_VARS if not os.getenv(name)]
    if missing:
        logger.debug("Azure OpenAI not configured; missing: %s", ", ".join(missing))
        return False
    return True


def _severity_to_decision(severity_text: str) -> str:
    """Map a severity label such as '3.0 Yellow card' to a card decision."""
    severity_text = (severity_text or "").strip()

    # Prefer the numeric severity prefix, which is unambiguous.
    prefix = severity_text.split(' ', 1)[0].rstrip('.')
    if prefix in SEVERITY_TO_DECISION:
        return SEVERITY_TO_DECISION[prefix]

    # Fall back to keyword matching for labels without a numeric prefix.
    lowered = severity_text.lower()
    if 'red' in lowered:
        return "Red Card"
    if 'yellow' in lowered:
        return "Yellow Card"
    if 'no card' in lowered:
        return "No Card"
    return "No Card"


def _format_predictions(predictions) -> str:
    """Render predictions as readable lines for the LLM prompt."""
    lines = []
    for pred in predictions:
        detail = pred.details[0]
        lines.append(
            f"{pred.category}: {detail.prediction} (confidence: {detail.probability:.1%})"
        )
    return "\n".join(lines)


def _rule_based_explanation(predictions) -> dict:
    """Deterministic fallback used when no LLM is configured or it fails."""
    action = "challenge"
    try_to_play = "No"
    touch_ball = "No"
    offence = ""
    decision = "No Card"

    for pred in predictions:
        category = pred.category
        value = pred.details[0].prediction
        if category == "Severity":
            decision = _severity_to_decision(value)
        elif category == "Action Class":
            action = value.lower()
        elif category == "Try to Play":
            try_to_play = value
        elif category == "Touch Ball":
            touch_ball = value
        elif category == "Offence":
            offence = value

    if decision == "No Card":
        # "No offence" means the incident was not a foul by the player we
        # assessed, which is the common case for a severity of 1.0.
        verdict = "The challenge was legal" if offence.lower() == "no offence" \
            else f"The {action} was legal"
        explanation = (
            f"{verdict}. The contact was not judged to be a foul, so play "
            f"continues and no card is warranted."
        )
        return {"decision": decision, "explanation": explanation}

    if try_to_play == "Yes":
        intent = "the defender was making a genuine attempt to play the ball"
    else:
        intent = "the defender was not making a genuine attempt to play the ball"

    if touch_ball == "Yes":
        outcome = "and still came away with the ball"
    else:
        outcome = "and denied the opponent a fair chance to reach the ball"

    if offence.lower() == "offence":
        lead = f"The {action} was a foul"
    else:
        lead = f"The {action} was judged to be a foul ({offence.lower()})"

    explanation = (
        f"{lead}: {intent} {outcome}. That is sanctionable as a "
        f"{decision.lower()}."
    )
    return {"decision": decision, "explanation": explanation}


def generate_explanation(predictions):
    """Generate a natural language explanation of the referee decision."""
    if not predictions:
        return {"decision": "No Card", "explanation": "No predictions were produced."}

    prediction_text = _format_predictions(predictions)

    if not _azure_configured():
        logger.info("Azure OpenAI is not configured; using the rule-based explanation.")
        return _rule_based_explanation(predictions)

    try:
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )

        system_prompt = (
            "You are a football referee expert. Provide a concise, natural-sounding "
            "explanation. Your response must be a valid JSON object with these fields:\n"
            "- decision: A concise decision (e.g., \"Yellow Card\", \"No Card\", \"Red Card\")\n"
            "- explanation: A natural, conversational explanation that sounds like a "
            "referee or commentator\n\n"
            "Keep the explanation concise but insightful, avoiding formulaic language. "
            "Use severity 1.0-2.0 for \"No Card\", 3.0-4.0 for \"Yellow Card\" and "
            "5.0 for \"Red Card\"."
        )

        user_prompt = (
            "Based on these predictions, explain the referee decision:\n\n"
            f"{prediction_text}\n\n"
            "Respond with a JSON object containing 'decision' and 'explanation' fields. "
            "Make your explanation sound natural and authoritative, like an experienced "
            "referee or commentator would explain it."
        )

        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=200,
            temperature=0.5,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Azure OpenAI returned an empty response")

        result = json.loads(content.strip())
        if "decision" not in result or "explanation" not in result:
            raise ValueError(
                f"Azure OpenAI response missing required fields: {content!r}"
            )

        return {
            "decision": str(result["decision"]),
            "explanation": str(result["explanation"]),
        }

    except Exception as exc:
        logger.error("Azure OpenAI explanation failed (%s); using rule-based fallback.", exc)
        return _rule_based_explanation(predictions)
