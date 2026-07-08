import json

def plural(word):
    if word.endswith("s"):
        return word
    return word + "s"

def object_phrase(obj_name, props):
    size = props.get("size")
    color = props.get("color")
    parts = []
    if size:
        parts.append(size)
    if color:
        parts.append(color)
    parts.append(obj_name)
    return "a " + " ".join(parts)

def order_to_text(order_dict):
    items = sorted(order_dict.items(), key=lambda x: x[1]["index"])
    phrases = []
    for name, props in items:
        # Original object name is the part before "_"
        obj_name = name.split("_")[0]
        phrase = object_phrase(obj_name, props)
        phrases.append(phrase)
    return ", ".join(phrases)

def count_to_text(count_dict):
    parts = []
    for obj_name, group in count_dict.items():
        for k, v in group.items():
            # normal mode -> key is "normal"; color mode -> key is the color name
            if k == "normal":
                num = v
                noun = plural(obj_name) if num > 1 else obj_name
                parts.append(f"{num} {noun}")
            else:
                # color mode
                num = v
                noun = plural(obj_name) if num > 1 else obj_name
                parts.append(f"{num} {k} {noun}")
    return ", ".join(parts)



def json2abstract(json_data) -> str:
    text_parts = []

    # ===== FRONT VIEW =====
    fv = json_data["front-view"]
    fv_order = order_to_text(fv["from-left-to-right"])
    text_parts.append(
        f"First, from the front side perspective (Can obtain the left-right relationship of objects), we can see from left to right, the order is {fv_order}."
    )

    # ===== LEFT VIEW =====
    lv = json_data["left-view"]
    lv_order = order_to_text(lv["from-left-to-right"])
    text_parts.append(
        f"Second, from the left side perspective (Can obtain the back-front relationship of objects), we can see from back to front, the order is {lv_order}."
    )

    # ===== TOP VIEW =====
    tv = json_data["top-view"]

    overall = count_to_text(tv["can-see"])
    text_parts.append(
        f"Finally, from the top side perspective (Can confirm its positional relationship both left-right and back-front), we can see from the overall perspective, there are {overall}, "
        f"we can see from left to right, the order is {order_to_text(tv['from-left-to-right'])}, "
        f"we can see from back to front, the order is {order_to_text(tv['from-top-to-down'])}."
    )

    return "\n".join(text_parts)
