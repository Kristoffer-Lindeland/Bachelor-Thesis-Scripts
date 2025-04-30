import cv2
import matplotlib.pyplot as plt

def show_pixel_difference(cover_path, stego_path):
    cover_color = cv2.imread(cover_path)
    stego_color = cv2.imread(stego_path)

    cover_rgb = cv2.cvtColor(cover_color, cv2.COLOR_BGR2RGB)
    stego_rgb = cv2.cvtColor(stego_color, cv2.COLOR_BGR2RGB)

    cover_gray = cv2.cvtColor(cover_color, cv2.COLOR_BGR2GRAY)
    stego_gray = cv2.cvtColor(stego_color, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(cover_gray, stego_gray)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title('Cover Image')
    plt.imshow(cover_rgb)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Stego Image')
    plt.imshow(stego_rgb)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Pixel Difference')
    plt.imshow(diff, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def count_length(number):
    return f"{number:06d}"

image_num = 46056
cover_image_path = f"C:/path/to/cover/image{count_length(image_num)}.jpg"
stego_image_path = f"C:/path/to/stego/image{count_length(image_num)}.jpg"

show_pixel_difference(cover_image_path, stego_image_path)