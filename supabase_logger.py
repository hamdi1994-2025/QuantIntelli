import logging
import json
import os
from typing import Optional, Dict, Any
import datetime
import re

def convert_numpy_floats(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_floats(elem) for elem in obj]
    elif isinstance(obj, float):
        return obj
    return obj

SUPABASE_PREDICTION_TABLE_NAME = os.getenv("SUPABASE_PREDICTION_TABLE_NAME")

def extract_simplified_contextual_outcome(analysis_response_text: str) -> Optional[str]:
    if not analysis_response_text:
        logging.warning("No analysis response text provided.")
        return None

    start = analysis_response_text.find("### EXECUTIVE_SUMMARY_START")
    end   = analysis_response_text.find("### EXECUTIVE_SUMMARY_END")
    if start != -1 and end != -1 and end > start:
        summary = analysis_response_text[start + len("### EXECUTIVE_SUMMARY_START"):end].strip()
    else:
        logging.info("No standard summary block found, using entire text.")
        summary = analysis_response_text

    extracted_value_from_source = None
    logging_source_info = ""

    for line in summary.splitlines():
        if "DUAL RECOMMENDATION" in line:
            m = re.search(r"\bOR\s+([^@|]+?)(?=\s*(?:@|\|))", line, re.IGNORECASE)
            if m:
                extracted_value_from_source = m.group(1).strip()
                logging_source_info = "DUAL RECOMMENDATION"
                break 
    
    if not extracted_value_from_source:
        for line in summary.splitlines():
            if "Preferred Outcome: Contextual" in line:
                m = re.search(r"Contextual\s*\(([^)]+)\)", line)
                if m:
                    raw_fallback_text = m.group(1).strip()
                    extracted_value_from_source = raw_fallback_text.split(" due to ")[0].split(" (")[0].strip()
                    logging_source_info = "Preferred Outcome: Contextual"
                    break 
    
    if not extracted_value_from_source:
        logging.warning("Could not extract any contextual outcome candidate.")
        return None

    # Normalize the extracted string for comparison (handles case and internal spaces)
    normalized_for_comparison = ' '.join(extracted_value_from_source.lower().split())

    if normalized_for_comparison == "home win":
        final_outcome = "Home"
        logging.info(f"Extracted and simplified contextual outcome from {logging_source_info}: '{final_outcome}' (original: '{extracted_value_from_source}')")
        return final_outcome
    elif normalized_for_comparison == "away win":
        final_outcome = "Away"
        logging.info(f"Extracted and simplified contextual outcome from {logging_source_info}: '{final_outcome}' (original: '{extracted_value_from_source}')")
        return final_outcome
    else:
        # No simplification for "Home Win" / "Away Win" applies. Return the extracted value as is.
        logging.info(f"Extracted contextual outcome from {logging_source_info}: '{extracted_value_from_source}' (no 'Win' rule simplification)")
        return extracted_value_from_source

def log_new_prediction_session(
    supabase_client,
    user_message_predict: str,
    prediction_context: Dict[str, Any],
    full_bot_response_predict: str
) -> Optional[str]:
    logging.info("Attempting to create new prediction session entry in Supabase...")

    if supabase_client is None:
        logging.warning("Supabase client not provided or initialized. Cannot create prediction session.")
        return None
    if not prediction_context or not prediction_context.get('odds') or not prediction_context.get('prediction') or 'probabilities' not in prediction_context:
         logging.error("Prediction context is incomplete or missing probabilities for saving.")
         return None

    try:
        odds = prediction_context['odds']
        teams = prediction_context.get('teams')
        pred_code = prediction_context['prediction']
        probabilities_data = prediction_context.get('probabilities', {})
        statistical_pred_str = {"W": "Home", "D": "Draw", "L": "Away"}.get(pred_code, pred_code)

        data_to_save = {
            "user_message_predict": user_message_predict,
            "match_teams": f"{teams[0]} - {teams[1]}" if teams and isinstance(teams, (list, tuple)) and len(teams) == 2 else None,
            "home_odds": odds.get('W'),
            "draw_odds": odds.get('D'),
            "away_odds": odds.get('L'),
            "statistical_prediction": statistical_pred_str,
            "statistical_probabilities": json.dumps(probabilities_data),
            "full_bot_response_predict": full_bot_response_predict,
            "contextual_prediction": None,
            "user_message_analyze": None,
            "full_bot_response_analyze": None,
            "updated_at": None
        }

        response = supabase_client.table(SUPABASE_PREDICTION_TABLE_NAME).insert([data_to_save]).execute()

        if response and hasattr(response, 'data') and response.data:
             new_id = response.data[0].get('id')
             logging.info(f"Successfully created prediction session entry. Record ID: {new_id}")
             return str(new_id)
        elif response and hasattr(response, 'error') and response.error:
             logging.error(f"Supabase insert failed for new session: {response.error.message if hasattr(response.error, 'message') else response.error}")
             return None
        else:
             logging.warning(f"Supabase insert for new session executed, but unexpected response format: {response}")
             return None

    except Exception as e:
        logging.exception(f"An unexpected error occurred during Supabase new session logging:")
        return None

def update_prediction_session_analysis(
    supabase_client,
    session_id: str,
    user_message_analyze: str,
    full_bot_response_analyze: str,
    prediction_context: Dict[str, Any]
) -> bool:
    logging.info(f"Attempting to update prediction session entry ID {session_id} with analysis...")

    if supabase_client is None:
        logging.warning("Supabase client not provided or initialized. Cannot update prediction session.")
        return False
    if not session_id:
        logging.error("No session_id provided for update.")
        return False

    try:
        contextual_outcome = extract_simplified_contextual_outcome(full_bot_response_analyze)
        
        if contextual_outcome is None:
            logging.warning("Could not extract a contextual outcome from the analysis. "
                           "Setting contextual_prediction to null in the database.")

        update_data = {
            "user_message_analyze": user_message_analyze,
            "contextual_prediction": contextual_outcome,
            "full_bot_response_analyze": full_bot_response_analyze,
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

        response = supabase_client.table(SUPABASE_PREDICTION_TABLE_NAME).update(update_data).eq('id', session_id).execute()

        if response and hasattr(response, 'data') and response.data:
             logging.info(f"Successfully updated prediction session entry ID {session_id} with analysis.")
             return True
        elif response and hasattr(response, 'error') and response.error:
             logging.error(f"Supabase update failed for session ID {session_id}: {response.error.message if hasattr(response.error, 'message') else response.error}")
             return False
        elif response and hasattr(response, 'count') and response.count > 0:
             logging.info(f"Successfully updated prediction session entry ID {session_id} (Count: {response.count}).")
             return True
        else:
             logging.warning(f"Supabase update for session ID {session_id} executed, but unexpected response format or no rows updated: {response}")
             return False

    except Exception as e:
        logging.exception(f"An unexpected error occurred during Supabase session update for ID {session_id}:")
        return False