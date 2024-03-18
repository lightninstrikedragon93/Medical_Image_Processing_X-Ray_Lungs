from PIL import Image
import os

def resize_images_in_folder(folder_path, output_folder, new_size=(224, 224)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            try:
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path)
                resized_img = img.resize(new_size)
                output_path = os.path.join(output_folder, filename)
                resized_img.save(output_path)
                print(f"Resized {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


input_folder_path = input('Enter input folder: ')
output_folder_path = input ('Enter output folder: ')
resize_images_in_folder(input_folder_path, output_folder_path)

