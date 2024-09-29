import sys
import argparse
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import re
import datetime
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

def save_image_with_timestamp(image, prefix, directory):
    ensure_directory_exists(directory)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = os.path.join(directory, filename)
    
    # Convert image to RGB mode if it's not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
        logging.info(f"Converted image from {image.mode} mode to RGB mode")
    
    image.save(filepath)
    logging.info(f"Saved image: {filepath}")
    return filepath

def process_image_question(image_path, question, output_directory):
    logging.info("Loading image and models...")
    image = Image.open(image_path)
    
    # Save original image with timestamp
    original_filename = save_image_with_timestamp(image, "original", output_directory)

    model_id = "allenai/Molmo-7B-D-0924"

    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype="auto"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype="auto", device_map=None
    )

    logging.info("Processing image and question...")
    inputs = processor.process(
        images=[image], 
        text=f"User: {question} Assistant:",
        stream="true"
    )

    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    logging.info("Generating output...")
    generation_config = GenerationConfig(max_new_tokens=100, stop_strings=["<|endoftext|>"])
    output = model.generate_from_batch(
        inputs, 
        generation_config=generation_config, 
        tokenizer=processor.tokenizer
    )

    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    logging.info("Processing generated output...")
    points = process_generated_output(generated_text, image.size)

    if points:
        logging.info("Drawing points on image...")
        draw_points_on_image(image, points)
        
        # Save modified image with timestamp
        modified_filename = save_image_with_timestamp(image, "modified", output_directory)
    else:
        logging.warning("No valid points found in the output.")

    return generated_text

def process_generated_output(generated_text, image_size):
    image_width, image_height = image_size
    points = []

    if '<point' in generated_text:
        match = re.search(r'<point x="([\d.]+)" y="([\d.]+)" alt=".*?">.*?</point>', generated_text)
        if match:
            points = [(float(match.group(1)), float(match.group(2)))]
    elif '<points' in generated_text:
        match = re.findall(r'x\d+="([\d.]+)"\s+y\d+="([\d.]+)"', generated_text)
        if match:
            points = [(float(x), float(y)) for x, y in match]

    absolute_points = [(int(x / 100 * image_width), int(y / 100 * image_height)) for x, y in points]
    return absolute_points

def draw_points_on_image(image, points):
    draw = ImageDraw.Draw(image)
    circle_radius = 10
    for x_abs, y_abs in points:
        circle_bbox = [x_abs - circle_radius, y_abs - circle_radius, x_abs + circle_radius, y_abs + circle_radius]
        draw.ellipse(circle_bbox, outline="pink", width=3, fill="pink")

def main():
    parser = argparse.ArgumentParser(description="Process an image and answer a question about it.")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("question", help="Question about the image")
    parser.add_argument("--output_dir", default="molmo_images", help="Directory to save output images")
    
    args = parser.parse_args()

    result = process_image_question(args.image_path, args.question, args.output_dir)
    print("Generated output:", result)

if __name__ == "__main__":
    main()