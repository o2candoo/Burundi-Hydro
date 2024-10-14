value1 = "Hallo"
value2 = "Welt"

# 20 Zeichen für value1 reservieren, danach value2
print("{:<20}{}".format(value1, value2))

# Alternativ mit f-strings
print(f"{value1:<20}{value2}")
