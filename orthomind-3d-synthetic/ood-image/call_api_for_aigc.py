import requests
import json
import os
import random
import re
import base64


def get_next_id(jsonl_path="./image_info.jsonl"):
    if not os.path.exists(jsonl_path):
        return 0
    
    max_id = -1
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    if data.get("id", -1) > max_id:
                        max_id = data["id"]
                except json.JSONDecodeError:
                    continue
    
    next_id = max_id + 1
    if next_id > 9999:
        raise ValueError("ID out of range (0-9999)")
    return next_id


def pluralize(word):
    irregulars = {
        "foot": "feet",
        "tooth": "teeth",
        "goose": "geese",
        "man": "men",
        "woman": "women",
        "child": "children",
        "mouse": "mice",
        "person": "people",
        "knife": "knives",
        "leaf": "leaves",
        "life": "lives",
        "wife": "wives",
    }
    
    unchanged = ["trousers", "headphones", "gloves", "scissors", "pants", "glasses"]
    
    if word in irregulars:
        return irregulars[word]
    
    if word in unchanged:
        return word
    
    if word.endswith(('s', 'x', 'z', 'ch', 'sh')):
        return word + "es"
    
    if word.endswith('y') and len(word) > 1 and word[-2] not in 'aeiou':
        return word[:-1] + "ies"
    
    o_es_words = ["potato", "tomato", "hero", "echo"]
    if word in o_es_words:
        return word + "es"
    
    if word.endswith('f') and not word.endswith('ff'):
        exceptions = ['roof', 'cliff', 'chief', 'belief', 'proof']
        if word not in exceptions:
            return word[:-1] + "ves"
    
    if word.endswith('fe'):
        return word[:-2] + "ves"
    
    return word + "s"

def sample_random_mode(num_types_to_select: int, power: float = 1.5) -> int:
    t = (num_types_to_select - 2) / 4
    p_mode_1 = 1 - (t ** power)
    return 1 if random.random() < p_mode_1 else 0

def generate_distinct_scene_prompt():
    items = [
        "apple", "banana", "orange", "water bottle", "gloves",
        "coke can", "book", "box", "headphones", "trousers",
        "clock", "trash can", "plant", "bag", "hat",
        "shoe", "watch", "wallet", "phone", "laptop",
        "tablet", "keyboard"
    ]
    
    rooms = [
        "vast empty modern living room floor with light wood texture",
        "large clean white gallery floor",
        "spacious minimalist concrete studio floor",
        "huge marble hall floor with no furniture",
        "wide open wooden deck surface"
    ]
    
    colors = ["red", "blue", "green", "yellow", "white", "black", "purple", "orange", "pink"]
    sizes = ["large", "small", "tiny", "giant", "wide", "tall"]
    other_mods = ["shiny", "matte", "vintage", "new", "old", "clean", "dusty"]
    
    num_types_to_select = random.randint(2, 6)
    selected_types = random.sample(items, num_types_to_select)
    
    prompt_parts = []
    objects_info = []
    
    for item_type in selected_types:
        if num_types_to_select > 3:
           weight_for_one = num_types_to_select - 3
           weights = [1 + weight_for_one, 1, 1, 1, 1]
           count = random.choices([1, 2, 3, 4, 5], weights=weights)[0]
        else:
          count = random.randint(1, 5)
        
        random_mode = sample_random_mode(num_types_to_select)

        if random_mode == 0:
            if count > 1:
                item_name_plural = pluralize(item_type)
                prompt_parts.append(f"{count} {item_name_plural}")
            else:
                prompt_parts.append(f"1 {item_type}")
        else:
            available_mods = list(colors)
            
            item_lower = item_type.lower()
            if any(c in item_lower for c in colors):
                available_mods = list(sizes) + ["open", "closed", "damaged", "clean"]
            
            while len(available_mods) < count:
                available_mods.extend(other_mods)
                available_mods.extend(sizes)
                available_mods = list(set(available_mods))
            
            chosen_mods = random.sample(available_mods, count)
            
            for mod in chosen_mods:
                prompt_parts.append(f"1 {mod} {item_type}")
        
        objects_info.append({"name": item_type, "count": count})
    
    random.shuffle(prompt_parts)
    
    if len(prompt_parts) > 1:
        items_str = ", ".join(prompt_parts[:-1]) + " and " + prompt_parts[-1]
    else:
        items_str = prompt_parts[0]
    
    chosen_room = random.choice(rooms)
    
    prompt = (
        f"A wide-angle photorealistic 3D render of {items_str} haphazardly scattered across a {chosen_room}. "
        "The scene features significant negative space around objects. "
        "Items are placed on the same horizontal plane (floor) but with random orientations. "
        "Each object is strictly isolated, no overlapping, no touching, widely spaced apart. "
    )
    
    if random_mode == 1:
        prompt += "Every single object must be distinct in visual appearance (color or size) to be easily identifiable. "
    
    prompt += "High angle shot, depth of field, soft volumetric lighting, Octane render, 8k resolution, ultra-detailed."
    
    return prompt, objects_info, random_mode

def extract_image_data(content):
    """
    Extract image data from content, support URL and base64 encoding
    Return: (type, data) tuple, type is 'url' or 'base64', data is the corresponding data
    """
    if not content:
        return None, None
    
    # First try to extract base64 encoded image
    # Match format: ![image](data:image/xxx;base64,xxxxx) or [image](data:image/xxx;base64,xxxxx)
    base64_match = re.search(r'!\[.*?\]\(data:image/([^;]+);base64,([^\s\)]+)\)', content)
    if not base64_match:
        base64_match = re.search(r'\[.*?\]\(data:image/([^;]+);base64,([^\s\)]+)\)', content)
    
    if base64_match:
        image_format = base64_match.group(1)
        base64_data = base64_match.group(2)
        return 'base64', (image_format, base64_data)
    
    # If no base64, try to extract URL
    match = re.search(r'!\[.*?\]\((https?://[^\s\)]+)\)', content)
    if match:
        return 'url', match.group(1)
    
    match = re.search(r'\[.*?\]\((https?://[^\s\)]+)\)', content)
    if match:
        return 'url', match.group(1)
    
    match = re.search(r'(https?://[^\s\)\"\'\]]+\.(?:png|jpg|jpeg|gif|webp|bmp))', content, re.IGNORECASE)
    if match:
        return 'url', match.group(1)
    
    match = re.search(r'(https?://[^\s\)\"\'\]]+)', content)
    if match:
        return 'url', match.group(1)
    
    return None, None


def generate_image(prompt, objects_info, image_id, api_key):
    url = os.environ.get("API_BASE_URL", "")
    
    payload = json.dumps({
        "stream": False,
        "model": "gemini-3-pro-image-preview",
        "messages": [
            {
                "content": prompt,
                "role": "user"
            }
        ]
    })
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    try:
        print(f"[ID {image_id}] Generating image...")
        response = requests.post(url, headers=headers, data=payload, timeout=180)
        response.raise_for_status()
        
        result = response.json()
        
        if "choices" not in result or len(result["choices"]) == 0:
            print(f"[ID {image_id}] API response format error: {result}")
            return False
        
        content = result["choices"][0]["message"]["content"]
        
        image_type, image_data = extract_image_data(content)
        if not image_type or not image_data:
            print(f"[ID {image_id}] Cannot extract image data from response (URL or base64)")
            print(f"Response content: {content[:500]}...")
            return False
        
        os.makedirs("./images", exist_ok=True)
        image_path = f"./images/{image_id}.png"
        
        if image_type == 'base64':
            # Handle base64 encoded image
            image_format, base64_data = image_data
            print(f"[ID {image_id}] Detected base64 encoded image, format: {image_format}")
            
            try:
                image_bytes = base64.b64decode(base64_data)
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                print(f"[ID {image_id}] Image saved: {image_path}")
            except Exception as e:
                print(f"[ID {image_id}] base64 decoding failed: {e}")
                return False
            
            # Do not save url field in base64 case
            record = {
                "id": image_id,
                "prompt": prompt,
                "objects": objects_info
            }
        else:
            # Handle URL image
            image_url = image_data
            print(f"[ID {image_id}] Image URL: {image_url}")
            
            print(f"[ID {image_id}] Downloading image...")
            img_response = requests.get(image_url, timeout=60)
            img_response.raise_for_status()
            
            content_type = img_response.headers.get('Content-Type', '')
            if not content_type.startswith('image/') and len(img_response.content) < 1000:
                print(f"[ID {image_id}] The downloaded content may not be a valid image: {content_type}")
                return False
            
            with open(image_path, "wb") as f:
                f.write(img_response.content)
            
            print(f"[ID {image_id}] Image saved: {image_path}")
            
            record = {
                "id": image_id,
                "url": image_url,
                "prompt": prompt,
                "objects": objects_info
            }
        
        with open("./image_info.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        print(f"[ID {image_id}] Metadata recorded to image_info.jsonl")
        return True
        
    except requests.exceptions.Timeout:
        print(f"[ID {image_id}] Request timeout")
        return False
    except requests.exceptions.RequestException as e:
        print(f"[ID {image_id}] Network request failed: {e}")
        return False
    except (KeyError, IndexError) as e:
        print(f"[ID {image_id}] Parsing response failed: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"[ID {image_id}] JSON parsing failed: {e}")
        return False
    except IOError as e:
        print(f"[ID {image_id}] File writing failed: {e}")
        return False


def main():
    API_KEY = os.environ.get("API_KEY")
    API_BASE_URL = os.environ.get("API_BASE_URL")
    if not API_KEY or not API_BASE_URL:
        print("Error: set the API_KEY and API_BASE_URL environment variables before running.")
        return

    os.makedirs("./images", exist_ok=True)
    
    try:
        start_id = get_next_id()
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    print(f"Start ID: {start_id}")
    
    num_images = 200
    success_count = 0
    fail_count = 0
    
    for i in range(num_images):
        current_id = start_id + i
        
        if current_id > 9999:
            print(f"ID {current_id} out of range, stop generating")
            break
        
        # random_mode = random.randint(0, 1)
        
        prompt, objects_info, random_mode = generate_distinct_scene_prompt()
        
        print(f"\n{'='*60}")
        print(f"[ID {current_id}] Generating mode: {'Detailed mode' if random_mode == 1 else 'Simple mode'}")
        print(f"Object list: {objects_info}")
        print(f"Prompt preview: {prompt[:150]}...")
        
        # Generate and save image
        success = generate_image(prompt, objects_info, current_id, API_KEY)
        
        if success:
            success_count += 1
            print(f"[ID {current_id}] ✓ Generating success")
        else:
            fail_count += 1
            print(f"[ID {current_id}] ✗ Generating failed")
    
    print(f"\n{'='*60}")
    print(f"Generating completed! Success: {success_count}, Failed: {fail_count}")


if __name__ == "__main__":
    main()