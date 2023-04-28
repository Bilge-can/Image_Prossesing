import cv2
import numpy as np

image = cv2.imread("cat.jpeg")

# Filtre matrislerini tanımlayın
filter1 = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])  # Sobel edge detection filter (yatay)

filter2 = np.array([[0, 1, 2],
                   [-1, 0, 1],
                   [-2, -1, 0]])  # Sobel edge detection filter (dikey)


def apply_filter(image, kernel):
    height, width, channels = image.shape
    k_size = kernel.shape[0]

    output_image = np.zeros_like(image)

    for y in range(height - k_size + 1):
        for x in range(width - k_size + 1):
            for c in range(channels):
                output_image[y, x, c] = np.sum(
                    kernel * image[y:y + k_size, x:x + k_size, c])

    return output_image

def max_pooling(image, pool_size):
    height, width = image.shape[:2]
    output_height = int(np.ceil(height / pool_size))
    output_width = int(np.ceil(width / pool_size))

    output_image = np.zeros((output_height, output_width, image.shape[2]))

    for y in range(0, height, pool_size):
        for x in range(0, width, pool_size):
            for c in range(image.shape[2]):
                output_image[y // pool_size, x // pool_size, c] = np.max(
                    image[y: y + pool_size, x: x + pool_size, c]
                )

    return output_image

def flatten_image(image):
    flattened_image = []
    for i in range(image.shape[0]):  # Satırlar
        for j in range(image.shape[1]):  # Sütunlar
            for k in range(image.shape[2]):  # Renk kanalları (RGB)
                flattened_image.append(image[i][j][k])
    return flattened_image

def show_images(images, titles):
    for i, (image, title) in enumerate(zip(images, titles)):
        cv2.imshow(title, image)

    # Kullanıcı bir tuşa basana kadar bekleyin
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Filtreleri uygulayın
filtered_image1 = apply_filter(image, filter1)
filtered_image2 = apply_filter(image, filter2)

# Max pooling işlemi uygulayın
pooled_image1 = max_pooling(filtered_image1, 2)
pooled_image2 = max_pooling(filtered_image2, 2)


# Matrisleri 1 boyutlu dizilere dönüştürün
flatten_img1 = flatten_image(pooled_image1)
flatten_img2 = flatten_image(pooled_image2)

print("flat:", flatten_img1[:20])

# Giriş resmi, filtrelenmiş resimler ve max pooled resimlerin listesini oluşturun
images = [image, filtered_image1,
          filtered_image2, pooled_image1, pooled_image2]
titles = ["Orjinal Resim", "Filtre 1",
          "Filtre 2", "Max Pooling 1", "Max Pooling 2"]

# Resimleri ekranda göster
show_images(images, titles)