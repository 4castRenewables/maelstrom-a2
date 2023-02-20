import re


def is_emoji(string):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U00002500-\U00002BEF"  # chinese char + emojis?
        "]+",
        re.UNICODE,
    )
    return bool(re.findall(emoji_pattern, string))


def get_emoji_filename(emoji):
    if len(emoji) == 1:
        return f"{ord(emoji):x}.png"
    elif len(emoji) > 1:
        out = ""
        for c in emoji:
            out += f"{ord(c):x}_"
        return f"{out[:-1]}.png"
