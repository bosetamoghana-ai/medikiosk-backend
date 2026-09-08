import re

def scan_for_red_flags(user_input: str):
    """
    Scans patient input for a comprehensive list of critical medical emergencies based on clinical guidelines.
    Uses regex word boundaries to prevent false positives and handles negations.
    """
    text = user_input.lower()
    
    # 1. Handle Negations (e.g., "no chest pain", "deny shortness of breath")
    negation_patterns = [
        r"\b(no|not|without|deny|denies|zero)\b.*?\b(chest pain|chest pressure|bleeding|breathlessness|fever|pain|weakness)\b",
        r"\b(don't|do not|doesn't)\b.*?\b(have|feel|experience)\b.*?\b(chest pain|bleeding|nausea|confusion)\b"
    ]
    for pattern in negation_patterns:
        if re.search(pattern, text):
            return None 

    # 2. Heart Attack (Acute MI) & Cardiac Arrest
    if re.search(r"\b(chest pressure|chest pain)\b", text):
        mi_markers = r"\b(sweating|nausea|breathlessness|arm|jaw|back)\b"
        if re.search(mi_markers, text):
            return {
                "category": "Heart Attack (Acute MI)",
                "matched_trigger": "Chest pressure/pain with radiating symptoms or autonomic signs",
                "patient_instruction": "Please seek immediate emergency medical care or call an ambulance. These are warning signs of a heart attack."
            }
    if re.search(r"\b(unresponsive|not breathing normally)\b", text):
        return {
            "category": "Cardiac Arrest",
            "matched_trigger": "Unresponsive and not breathing normally",
            "patient_instruction": "Call emergency services immediately and begin CPR/AED if trained."
        }

    # 3. Stroke & Neurological Emergencies
    stroke_patterns = r"\b(facial drooping|arm weakness|leg weakness|speech difficulty|sudden confusion|vision loss)\b"
    if re.search(stroke_patterns, text):
        return {
            "category": "Stroke",
            "matched_trigger": "Sudden facial drooping, weakness, or speech/vision difficulty",
            "patient_instruction": "These are signs of a potential stroke. Please have someone take you to the ER or call emergency services immediately."
        }

    # 4. Severe Breathing Difficulty & Anaphylaxis
    resp_patterns = r"\b(extreme breathlessness|blue lips|inability to speak|very low oxygen)\b"
    if re.search(resp_patterns, text):
        return {
            "category": "Severe Breathing Difficulty",
            "matched_trigger": "Extreme breathlessness or blue lips",
            "patient_instruction": "This is a respiratory emergency. Call for emergency medical assistance immediately."
        }
    anaphylaxis_patterns = r"\b(throat swelling|tongue swelling|wheezing|collapse)\b"
    if re.search(anaphylaxis_patterns, text) and re.search(r"\b(difficulty breathing|allergic)\b", text):
        return {
            "category": "Severe Allergic Reaction (Anaphylaxis)",
            "matched_trigger": "Airway swelling and difficulty breathing",
            "patient_instruction": "This is a life-threatening allergic reaction. Use an EpiPen immediately if prescribed and call emergency services."
        }

    # 5. Severe Bleeding, Trauma & Shock
    bleeding_patterns = r"\b(uncontrolled bleeding|vomiting blood|coughing blood|blood loss.*fainting)\b"
    if re.search(bleeding_patterns, text):
        return {
            "category": "Severe Bleeding",
            "matched_trigger": "Uncontrolled bleeding or internal bleeding signs",
            "patient_instruction": "Apply direct pressure to external wounds and proceed to the nearest Emergency Room immediately."
        }
    trauma_patterns = r"\b(serious head injury|serious chest injury|serious abdominal injury|spinal injury|major fractures?)\b"
    if re.search(trauma_patterns, text):
        return {
            "category": "Major Trauma",
            "matched_trigger": "Serious injury to head/chest/abdomen or suspected spinal injury",
            "patient_instruction": "Do not move unnecessarily, especially if a spinal injury is suspected. Call emergency services immediately."
        }
    shock_patterns = r"\b(very low blood pressure|fainting.*cold.*clammy skin|rapid pulse.*confusion)\b"
    if re.search(shock_patterns, text):
        return {
            "category": "Shock",
            "matched_trigger": "Signs of inadequate blood flow to vital organs",
            "patient_instruction": "This indicates clinical shock. Seek emergency medical attention immediately."
        }

    # 6. Major Burns & Poisoning
    burn_patterns = r"\b(extensive burns|facial burns|airway burns|electrical burns|chemical burns)\b"
    if re.search(burn_patterns, text):
        return {
            "category": "Major Burns",
            "matched_trigger": "Extensive, facial, electrical, or chemical burns",
            "patient_instruction": "Major burns carry a risk of airway injury and shock. Call emergency services."
        }
    tox_patterns = r"\b(suspected poisoning|overdose|altered consciousness.*suspected poisoning)\b"
    if re.search(tox_patterns, text):
        return {
            "category": "Poisoning/Overdose",
            "matched_trigger": "Suspected poisoning or overdose",
            "patient_instruction": "Contact emergency services or Poison Control immediately."
        }

    # 7. Diabetic, Metabolic & Seizure Emergencies
    if re.search(r"\b(seizure|seizures)\b", text):
        seizure_markers = r"\b(lasting \d+ minutes|repeated|without recovery|first severe|abnormal behavior|unconsciousness)\b"
        if re.search(seizure_markers, text):
            return {
                "category": "Seizure / Hypoglycemia Emergency",
                "matched_trigger": "Prolonged/repeated seizures or severe hypoglycemia",
                "patient_instruction": "This requires immediate clinical evaluation to prevent brain injury. Call emergency services."
            }
    diabetic_patterns = r"\b(severe vomiting.*dehydration|deep breathing.*confusion|rapid breathing.*confusion)\b"
    if re.search(diabetic_patterns, text):
        return {
            "category": "Diabetic Emergencies",
            "matched_trigger": "Signs of DKA or metabolic problems",
            "patient_instruction": "These symptoms may indicate a life-threatening metabolic problem. Proceed to the ER."
        }

    # 8. Sepsis & Meningitis (Infectious Emergencies)
    if re.search(r"\b(fever|low temperature)\b", text):
        sepsis_meningitis_patterns = r"\b(confusion|very fast breathing|very fast heart rate|extreme weakness|severe headache|neck stiffness)\b"
        if re.search(sepsis_meningitis_patterns, text):
            return {
                "category": "Severe Infection / Sepsis / Meningitis",
                "matched_trigger": "Fever coupled with neurological signs or extreme vitals",
                "patient_instruction": "A fever accompanied by these symptoms can progress rapidly to organ failure or brain damage. See an emergency doctor immediately."
            }

    # 9. Acute Abdomen, Pregnancy Complications & Pulmonary Embolism
    pregnancy_patterns = r"\b(heavy bleeding|severe abdominal pain|severe headache.*vision changes)\b"
    if re.search(pregnancy_patterns, text) and re.search(r"\b(pregnant|pregnancy)\b", text):
        return {
            "category": "Pregnancy Emergencies",
            "matched_trigger": "Pregnancy with heavy bleeding or preeclampsia signs",
            "patient_instruction": "This can threaten both mother and fetus. Proceed directly to the Emergency or Maternity triage desk."
        }
    abdomen_patterns = r"\b(severe persistent pain|rigid abdomen)\b"
    if re.search(abdomen_patterns, text):
        return {
            "category": "Sudden Severe Abdominal Pain",
            "matched_trigger": "Severe persistent pain or rigid abdomen",
            "patient_instruction": "This may indicate internal bleeding or a surgical emergency. Proceed to the ER."
        }
    pe_patterns = r"\b(sudden breathlessness.*chest pain|sudden breathlessness.*coughing blood)\b"
    if re.search(pe_patterns, text):
        return {
            "category": "Pulmonary Embolism",
            "matched_trigger": "Sudden breathlessness with chest pain or coughing blood",
            "patient_instruction": "These are warning signs of a blood clot in the lungs. Call emergency services immediately."
        }

    # Return None if no high-risk conditions are matched
    return None