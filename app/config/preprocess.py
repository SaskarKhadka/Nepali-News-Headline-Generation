import string, re

def perform_preprocessing(text):
    text = (
        text.replace("\n", " ")
        .replace("\r", " ")
        .replace("\t", " ")
        .replace("–", " ")
        .replace("-", " ")
        .replace("।", " ")
        # .replace("(", " ")
        # .replace(")", " ")
        .replace("—", " ")
        .replace("ॽ", " ")
        .replace("?", " ")
        .replace("ः", "")
        .replace(":", " ")
        .replace("“", ' ')
        .replace("”", ' ')
        .replace("‘", " ")
        .replace("’", " ")
        .replace('"', " ")
        .replace("'", " ")
        .replace("0", "०")
        .replace("1", "१")
        .replace("2", "२")
        .replace("3", "३")
        .replace("4", "४")
        .replace("5", "५")
        .replace("6", "६")
        .replace("7", "७")
        .replace("8", "८")
        .replace("9", "९")
    )

    # Remove emojis
    regrex_pattern = re.compile(
        pattern="["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags (iOS)
        "\U00002500-\U00002bef"  # chinese char
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2b55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+",
        flags=re.UNICODE,
    )
    text = regrex_pattern.sub("", text)

    pattern = re.compile(r'<.*?>') # HTML
    text = pattern.sub('', text)

    text = re.sub(r'([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})', '', text) # Email

    # Remove multiple white spaces
    text = re.sub(r"\s+", " ", text)

    text = re.sub('[a-zA-Z]', '', text)

    # Text inside paranthesis
    text = re.sub(r'\([^)]*\)', '', text)

    text = " ".join(text.split())

    punctuations = list(string.punctuation)
    punctuations.remove(".")
    punctuations.remove("%")
    # punctuations.remove("(")
    # punctuations.remove(")")

    text = "".join([char for char in text if char not in punctuations])
    text = " ".join(text.split())

    return text