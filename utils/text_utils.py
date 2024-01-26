def number_to_english(num):
    if not (1 <= num <= 200):
        return "输入数字应在1到200之间"

    # 定义数字对应的英文表示
    ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    teens = ["", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    if 1 <= num < 10:
        return ones[num]
    elif 10 < num < 20:
        return teens[num - 10]
    elif 20 <= num < 100:
        return tens[num // 10] + (" " + ones[num % 10] if num % 10 != 0 else "")
    elif 100 <= num < 1000:
        return ones[num // 100] + " hundred" + (" and " + number_to_english(num % 100) if num % 100 != 0 else "")
    elif 1000 <= num < 1000000:
        return number_to_english(num // 1000) + " thousand" + (" " + number_to_english(num % 1000) if num % 1000 != 0 else "")

