You'll want to pull the keys with:
dk_b_250, dk_a_250 (Rank DD) and dd_b2, dd_a2

"b" means unlearning split b and "a" means unlearning split a 

I use this function to match company names to the data:

def mention_detected(text, target_name):
    suffixes_to_remove = [
        'inc', 'corp', 'corporation', 'ltd', 'co', 'llc', 'group', 'plc', 'intl', 'incorporated',
        'holdings', 'limited', 'sa', 'nv', 'bv', 'ag', 'kg', 'gmbh', 'sarl', 'holding', 'international'
    ]
    target_name = target_name.lower()
    if "\"" in target_name:
        target_name = target_name.replace("\"", "")
    for suffix in suffixes_to_remove:
        target_name = target_name.replace(suffix, '').strip()
   
    option2 = target_name
    if "-" in target_name:
        option2 = option2.replace("-", " ")
   
    # Use word boundaries to match complete words only
    def has_word_match(text_lower, pattern):
        return bool(re.search(r'\b' + re.escape(pattern) + r'\b', text_lower))
   
    text_lower = text.lower()
    return (has_word_match(text_lower, target_name) or
            (option2 != target_name and has_word_match(text_lower, option2)))