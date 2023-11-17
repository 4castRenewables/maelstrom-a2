def string_to_unicode(string):
    return "".join(rf"\u{ord(chr):04X}" for chr in string)
