def detect_event(text):

    text = str(text).lower()

    if "profit" in text:
        return "Earnings"

    elif "merger" in text:
        return "Merger"

    elif "lawsuit" in text:
        return "Legal"

    else:
        return "General"